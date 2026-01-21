"""
Eval ReRanker Page

Deep-dive evaluation of a single query with a specific reranker.
Useful for understanding reranker behavior on specific examples.
"""

import streamlit as st
import json
import numpy as np
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATASETS,
    RE_RANKERS,
    DEFAULT_MANISCOPE_K,
    DEFAULT_MANISCOPE_ALPHA,
    DEFAULT_MANISCOPE_VERSION,
    MANISCOPE_VERSIONS,
    OPENROUTER_MODELS,
    OLLAMA_MODELS,
    DEFAULT_OPENROUTER_URL,
    DEFAULT_OLLAMA_URL,
    COLORS,
    PAGE_ICON,
    PAGE_LAYOUT
)
from utils.models import load_all_models, run_reranker, get_model_info, RerankerModelError
from utils.metrics import calculate_all_metrics
from utils.data_loader import load_mteb_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Eval ReRanker",
    page_icon="🔬",
    layout=PAGE_LAYOUT
)

@st.cache_data
def concat_all_docs(dataset_items):
    all_docs = []
    for item in dataset_items:
        all_docs += item["docs"]
    return all_docs

@st.cache_resource
def load_baseline_model(model_name='all-MiniLM-L6-v2'):
    """Load sentence transformer for baseline cosine similarity"""
    return SentenceTransformer(model_name)

def compute_baseline_scores(query: str, docs: list, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Compute baseline cosine similarity scores.

    Args:
        query: Query text
        docs: List of document texts
        model_name: Embedding model name (from Configuration)

    Returns:
        Numpy array of cosine similarity scores
    """
    model = load_baseline_model(model_name)
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(docs)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    return similarities

st.header("🔬 Evaluating ReRanker")
st.markdown("""
**Deep-dive evaluation** of a single query with a specific reranker.
Analyze reranker behavior, examine document scores, and understand ranking decisions.
""")

# ============================================================================
# Sidebar: Configuration
# ============================================================================

with st.sidebar:
    st.markdown("### 🎯 Evaluation Configuration")

    # ReRanker selection (single select)
    # st.markdown("#### ReRanker")
    selected_reranker = st.selectbox(
        "Select ReRanker",
        options=RE_RANKERS,
        help="Choose one reranker to evaluate"
    )


    # Maniscope configuration
    if selected_reranker == "Maniscope":
        st.markdown("#### Maniscope Settings")

        c__1, c__2 = st.columns(2)
        with c__1:
            maniscope_k = st.number_input(
                "k (neighbors)",
                min_value=3,
                max_value=15,
                value=DEFAULT_MANISCOPE_K,
                step=1,
                help="Number of neighbors for k-NN graph"
            )
            st.session_state['maniscope_k'] = maniscope_k

        with c__2:
            maniscope_alpha = st.number_input(
                "α (alpha)",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_MANISCOPE_ALPHA,
                step=0.1,
                help="0=pure geodesic, 1=pure cosine"
            )
            # Set in session state for load_all_models
            st.session_state['maniscope_alpha'] = maniscope_alpha

        # Version selection
        # st.markdown("#### Optimization Version")
        with st.expander("Engine Optimization", expanded=False):
            version_options = ['v0', 'v1', 'v2', 'v3', 'v2o']
            version_labels = {
                'v0': 'v0 - Baseline',
                'v1': 'v1 - GPU (3× faster)',
                'v2': 'v2 - Full Opt (5× faster)',
                'v3': 'v3 - Cached (1-10× variable)',
                'v2o': 'v2o - Ultimate ⭐ (20-235× faster)'
            }

            # Get default from config or session state
            default_version = st.session_state.get('maniscope_version', DEFAULT_MANISCOPE_VERSION)
            default_index = version_options.index(default_version) if default_version in version_options else 4

            maniscope_version = st.selectbox(
                "Version",
                options=version_options,
                format_func=lambda x: version_labels[x],
                index=default_index,
                help="Performance optimization level"
            )
            st.session_state['maniscope_version'] = maniscope_version

            # Show version details
            if maniscope_version in MANISCOPE_VERSIONS:
                version_info = MANISCOPE_VERSIONS[maniscope_version]
                st.info(f"**{version_info['name']}**\n\n{version_info['description']}\n\n"
                    f"Latency: {version_info['latency']} | Speedup: {version_info['speedup']}")

    # LLM configuration
    if selected_reranker == "LLM-Reranker":
        st.markdown("#### LLM Configuration")

        llm_provider = st.radio(
            "Provider",
            options=["OpenRouter", "Ollama"],
            horizontal=True,
            help="Choose LLM provider"
        )

        if llm_provider == "OpenRouter":
            openrouter_api_key = st.text_input(
                "API Key",
                type="password",
                value=os.getenv("OPENROUTER_API_KEY", ""),
                help="OpenRouter API key"
            )

            llm_model = st.selectbox(
                "Model",
                options=OPENROUTER_MODELS,
                help="Choose OpenRouter model"
            )

            llm_base_url = DEFAULT_OPENROUTER_URL
        else:
            llm_model = st.selectbox(
                "Model",
                options=OLLAMA_MODELS,
                help="Choose Ollama model"
            )

            llm_base_url = DEFAULT_OLLAMA_URL
            openrouter_api_key = None

        # Set in session state for load_all_models
        st.session_state['llm_model'] = llm_model
        st.session_state['llm_base_url'] = llm_base_url
        st.session_state['llm_api_key'] = openrouter_api_key if openrouter_api_key else "ollama"

    # st.markdown("---")

    # Dataset selection (single select)
    # st.markdown("#### Dataset")
    dataset_names = [d["name"] for d in DATASETS]
    selected_dataset_name = st.selectbox(
        "Select Dataset",
        options=dataset_names,
        help="Choose one dataset to evaluate on"
    )

    # Get selected dataset config
    selected_dataset = next(d for d in DATASETS if d["name"] == selected_dataset_name)

    st.markdown(f"**{selected_dataset['description']}**")
    st.markdown(f"📊 Total queries: {selected_dataset['queries']}")

    # Load dataset first (needed for both modes)
    try:
        dataset_path = Path(__file__).parent.parent.parent / "data" / selected_dataset["file"]
        dataset_items = load_mteb_dataset(dataset_path)

        if not dataset_items:
            st.error("No queries found in dataset")
            st.stop()

    except FileNotFoundError:
        st.error(f"Dataset file not found: {selected_dataset['file']}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

    # Query mode selection
    # st.markdown("#### Enter Query")
    query_mode = st.radio(
        "Query Mode",
        options=["From Dataset", "Custom Query"],
        horizontal=True,
        help="Choose a query from the dataset or enter your own"
    )

    # Create query options (used by both modes for document selection)
    query_options = {
        f"Query {i+1}: {item['query'][:60]}..." if len(item['query']) > 60 else f"Query {i+1}: {item['query']}": i
        for i, item in enumerate(dataset_items)
    }

    if query_mode == "From Dataset":
        # Select query from dataset
        selected_query_display = st.selectbox(
            "Select Query",
            options=list(query_options.keys()),
            help="Choose a query from the dataset"
        )

        query_idx = query_options.get(selected_query_display, -1)
        if query_idx > -1:
            selected_item = dataset_items[query_idx]
            query_text = selected_item['query']
            docs = selected_item['docs']
            relevance_map = selected_item['relevance_map']

    else:
        # Custom Query mode - enter custom query but use dataset documents
        user_query_text = st.text_input(
            "Enter Query",
            help="Enter a custom query to evaluate against dataset documents"
        )

        # Select which query's documents to use
        doc_source_display = st.selectbox(
            "Use Documents From",
            options=["__ALL_DOCS__"] + list(query_options.keys()),
            index=0,
            help="Select which query's documents to use as the candidate pool"
        )

        doc_source_idx = query_options.get(doc_source_display, -1)
        if doc_source_idx > -1:
            # Use documents from selected query, but query text is custom
            selected_item = dataset_items[doc_source_idx]
            query_text = user_query_text  # FIX: Use custom query, not dataset query
            docs = selected_item['docs']
            relevance_map = {}  # FIX: No ground truth for custom queries
        else:
            # Use all documents from dataset
            query_text = user_query_text
            docs = concat_all_docs(dataset_items)
            relevance_map = {}



    # Evaluate button
    evaluate_button = st.button(
        "🚀 Evaluate Query",
        type="primary",
        use_container_width=True,
        help="Run evaluation on the selected query"
    )

# ============================================================================
# Main Panel: Results Display
# ============================================================================

if evaluate_button:
    # Validation
    if not query_text:
        st.error("❌ Please enter a query")
        st.stop()

    if not docs or len(docs) == 0:
        st.error("❌ Please provide documents to rank")
        st.stop()

    # Get configured embedding model (from Configuration page)
    embedding_model = st.session_state.get('config_embedding_model', 'all-MiniLM-L6-v2')

    # Show configuration summary
    st.markdown("### 📋 Evaluation Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", selected_dataset["name"])
    with col2:
        st.metric("ReRanker", selected_reranker)
    with col3:
        st.metric("Documents", len(docs))
    with col4:
        st.metric("Embedding Model", embedding_model.split('/')[-1])

    # st.markdown("---")

    # Compute baseline scores first
    with st.spinner(f"Computing baseline (cosine similarity with {embedding_model.split('/')[-1]})..."):
        baseline_start = time.time()
        baseline_scores = compute_baseline_scores(query_text, docs, embedding_model)
        baseline_ranked_indices = np.argsort(baseline_scores)[::-1]
        baseline_latency_ms = (time.time() - baseline_start) * 1000

    # Calculate baseline metrics
    if relevance_map:
        baseline_rankings = baseline_ranked_indices.tolist()
        baseline_metrics = calculate_all_metrics(baseline_rankings, relevance_map)
    else:
        baseline_metrics = None

    # Load reranker (configuration already set in session state from sidebar)
    with st.spinner(f"Loading {selected_reranker}..."):
        try:
            rerankers = load_all_models([selected_reranker])
            reranker = rerankers[selected_reranker]
        except Exception as e:
            st.error(f"❌ Failed to load reranker: {str(e)}")
            st.stop()

    # Run reranking
    with st.spinner(f"Running {selected_reranker}..."):
        start_time = time.time()
        try:
            scores = run_reranker(reranker, query_text, docs)
            latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            st.error(f"❌ Reranking failed: {str(e)}")
            st.stop()

    # Get ranked indices
    ranked_indices = np.argsort(scores)[::-1]

    # Calculate metrics (if relevance available)
    if relevance_map:
        rankings = ranked_indices.tolist()
        metrics = calculate_all_metrics(rankings, relevance_map)
    else:
        metrics = None

    # Store results in session state (including baseline results)
    st.session_state['eval_results'] = {
        'query': query_text,
        'docs': docs,
        'scores': scores,
        'ranked_indices': ranked_indices,
        'relevance_map': relevance_map,
        'metrics': metrics,
        'latency_ms': latency_ms,
        'reranker': selected_reranker,
        'dataset': selected_dataset["name"],
        # Baseline results
        'baseline_scores': baseline_scores,
        'baseline_ranked_indices': baseline_ranked_indices,
        'baseline_latency_ms': baseline_latency_ms,
        'baseline_metrics': baseline_metrics
    }

    # Show completion message
    total_time = baseline_latency_ms + latency_ms
    st.success(f"✅ Evaluation complete! Total time: {total_time:.1f} ms ({total_time/1000:.2f}s)")
    st.info(f"📊 Baseline: {baseline_latency_ms:.1f}ms | {selected_reranker}: {latency_ms:.1f}ms")

# Display results if available
if 'eval_results' in st.session_state:
    results = st.session_state['eval_results']

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "📈 Score Distribution",
        "🔍 Detailed Analysis"
    ])

    # Tab 1: Overview - Baseline Comparison
    with tab1:
        st.markdown("### 🔬 Reranker vs Cosine Baseline Comparison")

        # Retrieve baseline results from session state (already computed during evaluation)
        baseline_scores = results['baseline_scores']
        baseline_ranked_indices = results['baseline_ranked_indices']
        baseline_latency_ms = results['baseline_latency_ms']
        baseline_metrics = results['baseline_metrics']

        # Query display
        st.markdown("#### Query")
        st.info(results['query'])

        # Summary metrics with separate latencies
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Documents", len(results['docs']))
        with col2:
            st.metric("Baseline Latency", f"{baseline_latency_ms:.1f} ms")
        with col3:
            st.metric(f"{results['reranker']} Latency", f"{results['latency_ms']:.1f} ms")
        with col4:
            total_latency_ms = baseline_latency_ms + results['latency_ms']
            st.metric("Total Latency", f"{total_latency_ms:.1f} ms",
                     help="Baseline + Reranker combined latency")
        with col5:
            if results['metrics'] and baseline_metrics:
                improvement = results['metrics'].get('MRR', 0) - baseline_metrics.get('MRR', 0)
                st.metric("MRR Improvement", f"{improvement:+.4f}")
            else:
                # Show speed comparison when no metrics available
                if results['latency_ms'] > 0 and baseline_latency_ms > 0:
                    ratio = baseline_latency_ms / results['latency_ms']
                    if ratio >= 1.0:
                        # Reranker is faster than baseline
                        st.metric("Speed vs Baseline", f"{ratio:.2f}× faster",
                                 delta=f"Baseline: {baseline_latency_ms:.1f}ms",
                                 delta_color="normal",
                                 help=f"{results['reranker']}: {results['latency_ms']:.1f}ms, Baseline: {baseline_latency_ms:.1f}ms")
                    else:
                        # Reranker is slower than baseline
                        slowdown = 1.0 / ratio
                        st.metric("Speed vs Baseline", f"{slowdown:.2f}× slower",
                                 delta=f"Baseline: {baseline_latency_ms:.1f}ms",
                                 delta_color="inverse",
                                 help=f"{results['reranker']}: {results['latency_ms']:.1f}ms, Baseline: {baseline_latency_ms:.1f}ms")

        # st.markdown("---")

        # Metrics Comparison Table
        if results['metrics'] and baseline_metrics:
            st.markdown("#### 📊 Metrics Comparison")

            metrics_comparison_data = {
                'Metric': ['MRR', 'NDCG@3', 'NDCG@10', 'P@1', 'P@3', 'MAP'],
                'Cosine Baseline': [
                    f"{baseline_metrics.get('MRR', 0):.4f}",
                    f"{baseline_metrics.get('NDCG@3', 0):.4f}",
                    f"{baseline_metrics.get('NDCG@10', 0):.4f}",
                    f"{baseline_metrics.get('P@1', 0):.4f}",
                    f"{baseline_metrics.get('P@3', 0):.4f}",
                    f"{baseline_metrics.get('MAP', 0):.4f}"
                ],
                f'{results["reranker"]}': [
                    f"{results['metrics'].get('MRR', 0):.4f}",
                    f"{results['metrics'].get('NDCG@3', 0):.4f}",
                    f"{results['metrics'].get('NDCG@10', 0):.4f}",
                    f"{results['metrics'].get('P@1', 0):.4f}",
                    f"{results['metrics'].get('P@3', 0):.4f}",
                    f"{results['metrics'].get('MAP', 0):.4f}"
                ],
                'Δ (Improvement)': [
                    f"{(results['metrics'].get('MRR', 0) - baseline_metrics.get('MRR', 0)):+.4f}",
                    f"{(results['metrics'].get('NDCG@3', 0) - baseline_metrics.get('NDCG@3', 0)):+.4f}",
                    f"{(results['metrics'].get('NDCG@10', 0) - baseline_metrics.get('NDCG@10', 0)):+.4f}",
                    f"{(results['metrics'].get('P@1', 0) - baseline_metrics.get('P@1', 0)):+.4f}",
                    f"{(results['metrics'].get('P@3', 0) - baseline_metrics.get('P@3', 0)):+.4f}",
                    f"{(results['metrics'].get('MAP', 0) - baseline_metrics.get('MAP', 0)):+.4f}"
                ]
            }

            metrics_comp_df = pd.DataFrame(metrics_comparison_data)

            # Style the improvement column
            def highlight_improvement(row):
                if row.name == len(row) - 1:  # Last column (improvement)
                    return [''] * len(row)
                styles = [''] * len(row)
                delta_val = float(row['Δ (Improvement)'])
                if delta_val > 0:
                    styles[-1] = 'background-color: #d4edda; color: #155724'  # Green
                elif delta_val < 0:
                    styles[-1] = 'background-color: #f8d7da; color: #721c24'  # Red
                return styles

            styled_metrics = metrics_comp_df.style.apply(highlight_improvement, axis=1)
            st.dataframe(styled_metrics, use_container_width=True, hide_index=True)
        elif results['metrics']:
            st.markdown("#### 📊 Reranker Metrics")
            metrics_df = pd.DataFrame([results['metrics']])
            st.dataframe(metrics_df, use_container_width=True)

        # st.markdown("---")

        # Rankings Comparison Table (Top-10)
        st.markdown("#### 🏆 Top-10 Rankings Comparison")

        has_relevance = bool(results['relevance_map'])

        # Category detection helper function
        def detect_category(doc_text):
            """Detect document category for AorB disambiguation tasks"""
            doc_lower = doc_text.lower()

            # Jaguar: Car vs Animal
            if any(keyword in doc_lower for keyword in ['sedan', 'suv', 'hp', 'suspension', 'sound system', 'meridian', 'f-pace', 'supercharged']):
                return '🚗 Car'
            elif any(keyword in doc_lower for keyword in ['bite force', 'predator', 'swimming', 'felid', 'apex', 'psi', 'prey']):
                return '🐆 Animal'

            # Python: Programming vs Snake
            elif any(keyword in doc_lower for keyword in ['exception', 'handling', 'code', 'syntax', 'programming', 'function', 'variable']):
                return '💻 Code'
            elif any(keyword in doc_lower for keyword in ['breeding', 'diet', 'habitat', 'species', 'reptile', 'wild']):
                return '🐍 Snake'

            # Apple: Company vs Fruit
            elif any(keyword in doc_lower for keyword in ['stock', 'nasdaq', 'market cap', 'cupertino', 'iphone', 'revenue']):
                return '📱 Company'
            elif any(keyword in doc_lower for keyword in ['orchard', 'tree', 'fertilizer', 'gala', 'harvest', 'fruit']):
                return '🍎 Fruit'

            # Java: Programming vs Coffee
            elif any(keyword in doc_lower for keyword in ['oop', 'class', 'object', 'jvm', 'virtual machine', 'programming language']):
                return '💻 Code'
            elif any(keyword in doc_lower for keyword in ['coffee', 'cultivation', 'bean', 'arabica', 'plantation']):
                return '☕ Coffee'

            # Mercury: Element vs Planet
            elif any(keyword in doc_lower for keyword in ['toxic', 'liquid metal', 'thermometer', 'element', 'poisoning']):
                return '⚗️ Element'
            elif any(keyword in doc_lower for keyword in ['orbit', 'solar system', 'crater', 'planet']):
                return '🪐 Planet'

            # Flow: Movement vs Psychology
            elif any(keyword in doc_lower for keyword in ['fluid', 'rate', 'velocity', 'stream', 'current']):
                return '💧 Movement'
            elif any(keyword in doc_lower for keyword in ['mental state', 'psychology', 'engagement', 'csikszentmihalyi']):
                return '🧠 Psychology'

            return ''  # No category detected

        # Build comparison table with doc IDs only
        ranking_data = []
        for rank in range(min(10, len(results['ranked_indices']))):
            reranker_idx = results['ranked_indices'][rank]
            baseline_idx = baseline_ranked_indices[rank]

            reranker_score = results['scores'][reranker_idx]
            baseline_score = baseline_scores[baseline_idx]

            # Detect categories
            baseline_category = detect_category(results['docs'][baseline_idx])
            reranker_category = detect_category(results['docs'][reranker_idx])

            row = {
                'Rank': rank + 1,
                'Baseline Doc ID': baseline_idx,
                'Category': baseline_category,
                'Baseline Score': f"{baseline_score:.4f}",
                f'{results["reranker"]} Doc ID': reranker_idx,
                'Category.1': reranker_category,
                f'{results["reranker"]} Score': f"{reranker_score:.4f}"
            }

            # Only add relevance column if we have ground truth
            if has_relevance:
                row['Baseline ✓'] = "✅" if str(baseline_idx) in results['relevance_map'] else ""
                row[f'{results["reranker"]} ✓'] = "✅" if str(reranker_idx) in results['relevance_map'] else ""

            ranking_data.append(row)

        ranking_comp_df = pd.DataFrame(ranking_data)

        # Highlight rows where reranking differs
        def highlight_difference(row):
            baseline_doc_idx = row['Baseline Doc ID']
            reranker_doc_idx = row[f'{results["reranker"]} Doc ID']

            if baseline_doc_idx != reranker_doc_idx:
                return ['background-color: #fff3cd'] * len(row)  # Yellow highlight
            return [''] * len(row)

        styled_rankings = ranking_comp_df.style.apply(highlight_difference, axis=1)
        st.dataframe(styled_rankings, use_container_width=True, hide_index=True, height=400)

        # Document Inspector Section
        # st.markdown("---")
        st.markdown("#### 📄 Document Inspector")
        st.caption("View full text of retrieved documents")

        # Collect unique doc IDs from both rankings (top-10)
        baseline_top10_ids = baseline_ranked_indices[:10].tolist()
        reranker_top10_ids = results['ranked_indices'][:10].tolist()
        all_doc_ids = sorted(set(baseline_top10_ids + reranker_top10_ids))

        # Create document table
        doc_inspection_data = []
        for doc_id in all_doc_ids:
            doc_text = results['docs'][doc_id]

            # Check where this doc appears
            baseline_rank = baseline_top10_ids.index(doc_id) + 1 if doc_id in baseline_top10_ids else "-"
            reranker_rank = reranker_top10_ids.index(doc_id) + 1 if doc_id in reranker_top10_ids else "-"

            # Get scores
            baseline_score = f"{baseline_scores[doc_id]:.4f}"
            reranker_score = f"{results['scores'][doc_id]:.4f}"

            relevant = "✅" if has_relevance and str(doc_id) in results['relevance_map'] else ""

            row = {
                'Doc ID': doc_id,
                'Baseline Rank': baseline_rank,
                f'{results["reranker"]} Rank': reranker_rank,
                'Baseline Score': baseline_score,
                f'{results["reranker"]} Score': reranker_score,
                'Document Text': doc_text
            }

            if has_relevance:
                row['✓'] = relevant

            doc_inspection_data.append(row)

        doc_inspection_df = pd.DataFrame(doc_inspection_data)

        # Display with expandable rows
        for idx, row in doc_inspection_df.iterrows():
            doc_id = row['Doc ID']
            baseline_rank = row['Baseline Rank']
            reranker_rank = row[f'{results["reranker"]} Rank']

            # Detect category for this document
            doc_category = detect_category(results['docs'][doc_id])

            # Create header with rank info
            header_parts = [f"**Doc {doc_id}**"]
            if doc_category:
                header_parts.append(doc_category)
            if baseline_rank != "-":
                header_parts.append(f"Baseline: #{baseline_rank}")
            if reranker_rank != "-":
                header_parts.append(f"{results['reranker']}: #{reranker_rank}")
            if has_relevance and row.get('✓') == "✅":
                header_parts.append("✅ Relevant")

            header = " | ".join(header_parts)

            with st.expander(header):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("**Baseline Score**")
                    st.write(row['Baseline Score'])
                with col2:
                    st.caption(f"**{results['reranker']} Score**")
                    st.write(row[f'{results["reranker"]} Score'])

                st.caption("**Document Text:**")
                st.markdown(row['Document Text'])

    # Tab 2: Score Distribution
    with tab2:
        st.markdown("### Score Distribution")

        # Create score distribution chart
        score_data = []
        for i, score in enumerate(results['scores']):
            is_relevant = str(i) in results['relevance_map']
            score_data.append({
                'Document Index': i,
                'Score': score,
                'Relevant': 'Relevant' if is_relevant else 'Non-Relevant'
            })

        score_df = pd.DataFrame(score_data)

        # Bar chart
        fig = px.bar(
            score_df,
            x='Document Index',
            y='Score',
            color='Relevant',
            title='Scores by Document',
            color_discrete_map={'Relevant': '#28a745', 'Non-Relevant': '#6c757d'}
        )

        fig.update_layout(
            xaxis_title="Document Index",
            yaxis_title="Score",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Score statistics
        st.markdown("#### Score Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Min Score", f"{np.min(results['scores']):.4f}")
        with col2:
            st.metric("Max Score", f"{np.max(results['scores']):.4f}")
        with col3:
            st.metric("Mean Score", f"{np.mean(results['scores']):.4f}")
        with col4:
            st.metric("Std Dev", f"{np.std(results['scores']):.4f}")

        # Histogram
        fig_hist = px.histogram(
            score_df,
            x='Score',
            nbins=20,
            title='Score Distribution (Histogram)',
            color='Relevant',
            color_discrete_map={'Relevant': '#28a745', 'Non-Relevant': '#6c757d'}
        )

        fig_hist.update_layout(
            xaxis_title="Score",
            yaxis_title="Count",
            height=400
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    # Tab 3: Detailed Analysis
    with tab3:
        st.markdown("### Detailed Analysis")

        # Top-3 and Bottom-3
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔝 Top-3 Documents")
            for i, idx in enumerate(results['ranked_indices'][:3]):
                doc = results['docs'][idx]
                score = results['scores'][idx]
                is_relevant = str(idx) in results['relevance_map']

                relevance_badge = "✅ Relevant" if is_relevant else "❌ Not Relevant"

                with st.expander(f"**Rank {i+1}** - {relevance_badge}"):
                    st.markdown(f"**Score:** {score:.4f}")
                    st.markdown(f"**Original Index:** {idx}")
                    st.markdown("**Document:**")
                    st.markdown(doc)

        with col2:
            st.markdown("#### 🔻 Bottom-3 Documents")
            for i, idx in enumerate(results['ranked_indices'][-3:]):
                doc = results['docs'][idx]
                score = results['scores'][idx]
                is_relevant = str(idx) in results['relevance_map']

                relevance_badge = "✅ Relevant" if is_relevant else "❌ Not Relevant"
                rank = len(results['ranked_indices']) - 2 + i

                with st.expander(f"**Rank {rank}** - {relevance_badge}"):
                    st.markdown(f"**Score:** {score:.4f}")
                    st.markdown(f"**Original Index:** {idx}")
                    st.markdown("**Document:**")
                    st.markdown(doc)

        # Relevant documents analysis
        if results['relevance_map']:
            # st.markdown("---")
            st.markdown("#### ✅ Relevant Documents Analysis")

            relevant_positions = []
            for doc_idx_str in results['relevance_map'].keys():
                doc_idx = int(doc_idx_str)
                # Find position in ranked list
                position = np.where(results['ranked_indices'] == doc_idx)[0][0] + 1
                score = results['scores'][doc_idx]
                relevant_positions.append({
                    'Document Index': doc_idx,
                    'Rank': position,
                    'Score': score
                })

            relevant_df = pd.DataFrame(relevant_positions)
            relevant_df = relevant_df.sort_values('Rank')

            st.dataframe(relevant_df, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Relevant", len(relevant_positions))
            with col2:
                best_rank = relevant_df['Rank'].min()
                st.metric("Best Rank", f"#{best_rank}")
            with col3:
                worst_rank = relevant_df['Rank'].max()
                st.metric("Worst Rank", f"#{worst_rank}")

else:
    # No results yet
    st.info("👈 Configure your evaluation in the sidebar and click **Evaluate Query** to start.")

    st.markdown("### About This Page")
    st.markdown("""
    This page allows you to perform deep-dive evaluation of a single query with a specific reranker.

    **Features:**
    - **Dataset queries**: Select from existing benchmark datasets
    - **Custom queries**: Enter your own query and documents
    - **Single reranker**: Focus on one reranker at a time
    - **Detailed analysis**: View ranked documents, score distributions, and more

    **Use Cases:**
    - Understand reranker behavior on specific examples
    - Debug ranking issues
    - Compare rerankers on challenging queries (from Analytics page)
    - Test custom queries
    """)
