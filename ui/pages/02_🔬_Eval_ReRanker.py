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
import hashlib

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
    """Load sentence transformer for baseline cosine similarity with robust CUDA fallback"""
    import torch
    import os

    # Force CPU for now if there are persistent CUDA issues
    force_cpu = os.getenv('MANISCOPE_FORCE_CPU', 'false').lower() == 'true'

    if force_cpu:
        st.info("💻 Using CPU mode (MANISCOPE_FORCE_CPU=true)")
        return SentenceTransformer(model_name, device='cpu')

    # Try CUDA with comprehensive error handling
    if torch.cuda.is_available():
        try:
            # Clear CUDA cache first
            torch.cuda.empty_cache()

            # Load model on CUDA
            model = SentenceTransformer(model_name, device='cuda')

            # Test with minimal computation to ensure CUDA works
            with torch.no_grad():
                _ = model.encode(["test"], show_progress_bar=False, convert_to_numpy=True)

            return model

        except (RuntimeError, Exception) as cuda_error:
            error_msg = str(cuda_error)
            if "CUDA" in error_msg or "GPU" in error_msg or "device" in error_msg.lower():
                st.warning(f"⚠️ GPU acceleration failed, using CPU mode for compatibility")
            else:
                st.warning(f"⚠️ Model loading failed, falling back to CPU")

            # Clear any CUDA state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # CPU fallback (always works)
    return SentenceTransformer(model_name, device='cpu')

def compute_baseline_scores(query: str, docs: list, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Compute baseline cosine similarity scores with error handling.

    Args:
        query: Query text
        docs: List of document texts
        model_name: Embedding model name (from Configuration)

    Returns:
        Numpy array of cosine similarity scores
    """
    try:
        model = load_baseline_model(model_name)

        # Compute embeddings with error handling
        query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        doc_embeddings = model.encode(docs, show_progress_bar=False, convert_to_numpy=True)

        # Compute similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        return similarities

    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "GPU" in error_msg:
            st.error(f"❌ GPU computation failed: {error_msg}")
            st.info("💡 Try setting environment variable MANISCOPE_FORCE_CPU=true to force CPU mode")
        else:
            st.error(f"❌ Embedding computation failed: {error_msg}")
        raise

def _get_cache_key(llm_model: str, query: str, context: str) -> str:
    """Generate cache key from model, query, and context."""
    # Create a unique hash from the combination
    cache_string = f"{llm_model}|{query}|{context}"
    return hashlib.sha256(cache_string.encode()).hexdigest()


def _load_from_cache(cache_key: str) -> str | None:
    """Load cached response if it exists."""
    cache_dir = Path(__file__).parent.parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"rag_{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return cached_data.get('response')
        except Exception:
            return None
    return None


def _save_to_cache(cache_key: str, response: str, llm_model: str, query: str):
    """Save response to cache."""
    cache_dir = Path(__file__).parent.parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"rag_{cache_key}.json"

    try:
        cache_data = {
            'response': response,
            'model': llm_model,
            'query': query,
            'timestamp': time.time()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception:
        pass  # Silently fail if cache write fails


def _cached_llm_call(llm_model: str, query: str, context: str, llm_provider: str, base_url: str, api_key: str) -> str:
    """
    Cached LLM API call. Uses disk cache in cache/ folder based on (model, query, context) key.

    Args:
        llm_model: LLM model identifier
        query: User query
        context: Document context (pre-formatted)
        llm_provider: Provider name
        base_url: API base URL
        api_key: API key

    Returns:
        Generated answer text
    """
    from openai import OpenAI

    # Check cache first
    cache_key = _get_cache_key(llm_model, query, context)
    cached_response = _load_from_cache(cache_key)

    if cached_response:
        st.caption("💾 Using cached response")
        return cached_response

    # Create prompt
    prompt = f"""Answer the following query using ONLY the information provided in the documents below. If the documents don't contain enough information to answer the query, say so.

Query: {query}

Documents:
{context}

Answer:"""

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Generate response
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500
    )

    content = response.choices[0].message.content
    result = content if content else "No response generated."

    # Save to cache
    _save_to_cache(cache_key, result, llm_model, query)

    return result


def generate_rag_response(query: str, documents: list, llm_model: str, llm_provider: str, api_key: str) -> str:
    """
    Generate RAG response using LLM with retrieved documents as context.
    Uses caching to avoid redundant API calls for same (model, query, context) combinations.

    Args:
        query: User query
        documents: List of retrieved document texts to use as context
        llm_model: LLM model identifier
        llm_provider: Provider name ("OpenRouter" or "Ollama")
        api_key: API key (for OpenRouter, falls back to OPENROUTER_API_KEY env var)

    Returns:
        Generated answer text
    """
    # Prepare context from documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"Document {i}:\n{doc}")

    context = "\n\n".join(context_parts)

    # Set up API client configuration
    if llm_provider == "OpenRouter":
        base_url = DEFAULT_OPENROUTER_URL

        # Use provided API key, or fall back to environment variable
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY", "")

        if not api_key:
            raise ValueError("OpenRouter API key is required. Please set it in Configuration page or set OPENROUTER_API_KEY environment variable.")
    else:  # Ollama
        base_url = DEFAULT_OLLAMA_URL
        api_key = "ollama"  # Ollama doesn't require real API key

    # Call cached LLM function
    # Cache key is based on: (llm_model, query, context, llm_provider, base_url, api_key)
    return _cached_llm_call(llm_model, query, context, llm_provider, base_url, api_key)

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

    # Dataset type selection
    use_custom_dataset = st.checkbox(
        "📂 Use Custom Dataset",
        value=False,
        help="Toggle to use custom datasets uploaded via Data Manager instead of BEIR datasets"
    )

    # Dataset selection based on type
    if use_custom_dataset:
        # Check for custom datasets in session state
        custom_datasets = []

        # Check for uploaded MTEB-format datasets
        if 'dataset' in st.session_state and 'dataset_source' in st.session_state:
            if st.session_state.get('dataset_source') in ['upload', 'custom']:
                dataset_name = st.session_state.get('dataset_name', 'Custom Dataset')
                dataset_data = st.session_state['dataset']
                num_queries = len(dataset_data) if dataset_data else 0
                total_docs = sum(len(item.get('docs', [])) for item in dataset_data) if dataset_data else 0

                custom_datasets.append({
                    'name': dataset_name,
                    'source': st.session_state['dataset_source'],
                    'queries': num_queries,
                    'total_docs': total_docs,
                    'data': dataset_data
                })

        # Check for custom datasets in data/custom/ directory
        custom_dir = Path(__file__).parent.parent.parent / "data" / "custom"
        if custom_dir.exists():
            for json_file in custom_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        custom_data = json.load(f)
                        if isinstance(custom_data, list) and custom_data:
                            # Calculate total documents across all items
                            total_docs = sum(len(item.get('docs', [])) for item in custom_data)
                            dataset_items_count = len(custom_data)

                            custom_datasets.append({
                                'name': f"📁 {json_file.stem}",
                                'source': 'file',
                                'queries': dataset_items_count,
                                'total_docs': total_docs,
                                'data': custom_data,
                                'file_path': json_file
                            })
                except (json.JSONDecodeError, Exception):
                    continue  # Skip invalid files

        if not custom_datasets:
            st.warning("⚠️ No custom datasets found. Upload a dataset via **Data Manager** page or uncheck this option to use BEIR datasets.")
            st.stop()

        # Custom dataset dropdown - clean names only
        custom_dataset_options = [ds['name'] for ds in custom_datasets]

        selected_custom_idx = st.selectbox(
            "Select Custom Dataset",
            options=range(len(custom_dataset_options)),
            format_func=lambda i: custom_dataset_options[i],
            help="Choose a custom dataset"
        )

        selected_custom = custom_datasets[selected_custom_idx]
        dataset_items = selected_custom['data']
        total_docs = selected_custom.get('total_docs', sum(len(item.get('docs', [])) for item in dataset_items))

        # Create a mock selected_dataset for compatibility with existing code
        selected_dataset = {
            'name': selected_custom['name'],
            'description': f"Custom dataset ({selected_custom['source']})",
            'queries': selected_custom['queries']
        }

        # Display info with clarified terminology
        st.markdown(f"**Custom Dataset:** {selected_custom['name']}")

        if selected_custom['queries'] == 1 and total_docs > 1:
            # PDF-style dataset
            st.markdown(f"📄 **PDF Import Dataset**: 1 dataset item containing {total_docs} text chunks")
            st.info("💡 This dataset was created from PDF import. In Custom Query mode, you can search across all text chunks.")
        else:
            # Standard multi-query dataset
            st.markdown(f"📊 **Dataset Items**: {selected_custom['queries']} queries with {total_docs} total documents")

        if selected_custom['source'] == 'custom':
            st.info("📝 No ground truth available - use Custom Query mode for exploration")

    else:
        # BEIR dataset selection (original logic)
        dataset_names = [d["name"] for d in DATASETS]
        selected_dataset_name = st.selectbox(
            "Select Dataset",
            options=dataset_names,
            help="Choose one BEIR dataset to evaluate on"
        )

        # Get selected dataset config
        selected_dataset = next(d for d in DATASETS if d["name"] == selected_dataset_name)

        st.markdown(f"**{selected_dataset['description']}**")
        st.markdown(f"📊 Total queries: {selected_dataset['queries']}")

        # Load BEIR dataset
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

    # Store dataset source info in session state for query mode logic
    if use_custom_dataset:
        st.session_state['current_dataset_source'] = 'custom'
    else:
        st.session_state['current_dataset_source'] = 'beir'

    # Query mode selection
    # Check if using custom dataset (may not have ground truth)
    current_dataset_source = st.session_state.get('current_dataset_source', 'beir')
    is_custom_dataset = current_dataset_source == 'custom'

    # Check if current dataset has ground truth (relevance_map with meaningful values)
    has_ground_truth = False
    if dataset_items and len(dataset_items) > 0:
        first_item = dataset_items[0]
        relevance_map = first_item.get('relevance_map', {})
        if relevance_map:
            # Check if relevance_map has non-zero values (indicating real ground truth)
            has_ground_truth = any(float(v) > 0 for v in relevance_map.values())

    if is_custom_dataset and not has_ground_truth:
        # Force Custom Query mode for custom datasets without ground truth
        st.info("📝 Custom datasets without ground truth use **Custom Query** mode")
        query_mode = "Custom Query"
    else:
        # Show radio button for datasets with ground truth
        query_mode = st.radio(
            "Query Mode",
            options=["From Dataset", "Custom Query"],
            horizontal=True,
            help="Choose a query from the dataset or enter your own"
        )

    # Initialize variables
    query_text = ""
    docs = []
    relevance_map = {}
    is_all_docs_mode = False

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
            is_all_docs_mode = False

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

        if doc_source_display == "__ALL_DOCS__":
            # Use all documents from dataset
            query_text = user_query_text
            docs = concat_all_docs(dataset_items)
            relevance_map = {}  # No ground truth for custom queries
            is_all_docs_mode = True

            with st.expander("About All Documents Mode", expanded=False):
                # Display corpus statistics
                total_queries = len(dataset_items)
                total_docs = len(docs)

                if total_queries == 1 and total_docs > 1:
                    # PDF-style dataset explanation
                    st.info("💡 **PDF Document Search**: Searching across all text chunks from the imported PDF document. Your custom query will be matched against all sections, paragraphs, and content from the original document.")
                    st.success(f"📄 **Document Statistics**: {total_docs:,} text chunks from PDF document - Full content search")
                else:
                    # Standard dataset explanation
                    avg_docs_per_query = total_docs / total_queries if total_queries > 0 else 0
                    st.info("💡 **RAG Performance Profiling Mode**: Using all documents from the entire dataset simulates a realistic RAG pipeline with large corpus retrieval. Perfect for stress testing and performance benchmarking!")
                    st.success(f"📊 **Corpus Statistics**: {total_docs:,} documents from {total_queries} queries (avg {avg_docs_per_query:.1f} docs/query) - Realistic RAG scenario")
        else:
            # Use documents from selected query
            doc_source_idx = query_options.get(doc_source_display, -1)
            if doc_source_idx > -1:
                selected_item = dataset_items[doc_source_idx]
                query_text = user_query_text  # Use custom query, not dataset query
                docs = selected_item['docs']
                relevance_map = {}  # No ground truth for custom queries
                is_all_docs_mode = False
            else:
                st.error("Invalid document source selected")
                st.stop()



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

    # Check if in Real RAG Simulation mode
    try:
        simulation_mode = is_all_docs_mode
    except NameError:
        simulation_mode = False

    if simulation_mode:
        st.info("🎯 **Real RAG Simulation Mode**: Profiling with full corpus - realistic performance benchmarking")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", selected_dataset["name"])
    with col2:
        st.metric("ReRanker", selected_reranker)
    with col3:
        st.metric("Documents", f"{len(docs):,}")
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

    # Check if we're in __ALL_DOCS__ mode (set in query selection logic above)
    # is_all_docs_mode is defined in the sidebar, but we need to access it here
    try:
        all_docs_mode = is_all_docs_mode
    except NameError:
        # Fallback if variable not set
        all_docs_mode = False

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
        'baseline_metrics': baseline_metrics,
        # RAG profiling mode
        'is_all_docs_mode': all_docs_mode,
        'num_docs': len(docs)
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

        # RAG Evaluation Section
        st.markdown("---")
        c1,_,c2 = st.columns([3,1,12])
        with c1:
            st.markdown("#### 🤖 RAG Evaluation")
            # st.caption("Compare RAG response quality using different reranking methods")
        with c2:
            # Add selector for which ranking method to use
            ranking_method = st.radio(
                "Use documents from:",
                options=["Baseline (Cosine Similarity)", f"{results['reranker']} (Reranker)"],
                horizontal=True,
                index=1,
                help="Select which document ranking to use for RAG context",
                key="rag_ranking_method"
            )

        # Get RAG configuration from session state
        rag_top_k = st.session_state.get('config_rag_top_k', 3)
        rag_llm_model = st.session_state.get('config_rag_llm_model', 'google/gemini-2.0-flash-lite-001')
        rag_llm_provider = st.session_state.get('config_rag_llm_provider', 'OpenRouter')

        # Get API key from session state, or fall back to environment variable
        rag_api_key = st.session_state.get('config_rag_api_key', '')
        if not rag_api_key:
            rag_api_key = os.getenv("OPENROUTER_API_KEY", "")

        # Determine which ranking to use
        if ranking_method == "Baseline (Cosine Similarity)":
            selected_ranked_indices = baseline_ranked_indices
            selected_scores = baseline_scores
            method_name = "Baseline"
        else:
            selected_ranked_indices = results['ranked_indices']
            selected_scores = results['scores']
            method_name = results['reranker']

        # Create two columns for comparison
        col_left, _, col_right = st.columns([20, 1, 16])

        # Left column: Display top-10 retrieved documents (Document Inspector style)
        with col_left:
            st.markdown(f"##### 📄 Top-10 Retrieved Documents ({method_name})")

            for rank in range(min(10, len(selected_ranked_indices))):
                doc_idx = selected_ranked_indices[rank]
                doc_text = results['docs'][doc_idx]
                score = selected_scores[doc_idx]

                # Check relevance
                is_relevant = str(doc_idx) in results['relevance_map'] if results['relevance_map'] else False
                relevance_badge = "✅" if is_relevant else ""

                # Detect category for this document
                doc_category = detect_category(results['docs'][doc_idx])

                # Create header with rank info
                header_parts = [f"**#{rank+1} - Doc {doc_idx}**"]
                if doc_category:
                    header_parts.append(doc_category)
                if is_relevant:
                    header_parts.append("✅ Relevant")
                header_parts.append(f"Score: {score:.4f}")

                header = " | ".join(header_parts)

                with st.expander(header, expanded=(rank in range(rag_top_k))):
                    st.markdown(doc_text)

        # Right column: Generate and display RAG response
        with col_right:
            st.markdown(f"##### 💬 RAG Response ({method_name} Context)")

            # Generate button and Top-K selector in one row
            col_btn,  col_k = st.columns([7, 3])
            with col_btn:
                st.caption(f"Using model: **{rag_llm_model}** ({rag_llm_provider})")
                generate_btn = st.button(
                    "🚀 Generate Answer",
                    key=f"generate_rag_{method_name}",
                    type="primary",
                    use_container_width=True
                )

            with col_k:
                current_top_k = st.number_input(
                    "Top-K",
                    min_value=1,
                    max_value=10,
                    value=rag_top_k,
                    step=1,
                    help="Number of top-ranked documents to use as context",
                    key=f"rag_top_k_{method_name}"
                )


            # Button to generate RAG response
            if generate_btn:
                # Get top-k documents
                top_k_indices = selected_ranked_indices[:current_top_k]
                top_k_docs = [results['docs'][idx] for idx in top_k_indices]

                # Generate RAG response
                with st.spinner("Generating RAG response..."):
                    try:
                        llm_start_time = time.time()
                        rag_response = generate_rag_response(
                            query=results['query'],
                            documents=top_k_docs,
                            llm_model=rag_llm_model,
                            llm_provider=rag_llm_provider,
                            api_key=rag_api_key
                        )
                        llm_latency_ms = (time.time() - llm_start_time) * 1000

                        # Store in session state
                        st.session_state[f'rag_response_{method_name}'] = rag_response
                        st.session_state[f'rag_llm_latency_{method_name}'] = llm_latency_ms

                    except Exception as e:
                        st.error(f"❌ Failed to generate RAG response: {str(e)}")

            # Display stored RAG response if available
            if f'rag_response_{method_name}' in st.session_state:
                st.markdown("**Query:**")
                st.info(results['query'])

                st.markdown("**Answer:**")
                st.success(st.session_state[f'rag_response_{method_name}'])

                # st.caption(f"Generated using top-{current_top_k} documents from {method_name}")

                # Display latency breakdown
                # st.markdown("---")
                is_rag_simulation = results.get('is_all_docs_mode', False)
                latency_section_label = f"⏱️ Latency Breakdown (Simulation Mode)" if is_rag_simulation else "⏱️ Latency Breakdown"
                with st.expander(latency_section_label, expanded=False):
                    # Check if we're in Real RAG Simulation mode
                    num_docs = results.get('num_docs', len(results['docs']))
                    if is_rag_simulation:
                        st.caption(f"Performance profiling with {num_docs:,} documents corpus")

                    # Get latency values based on selected method
                    if method_name == "Baseline":
                        # For baseline: only baseline latency, no separate reranking
                        global_retrieval_ms = results.get('baseline_latency_ms', 0)
                        local_reranking_ms = 0
                        reranking_label = "Local Reranking"
                        reranking_help = "No reranking (using baseline only)"
                    else:
                        # For reranker: baseline + reranker latency
                        global_retrieval_ms = results.get('baseline_latency_ms', 0)
                        local_reranking_ms = results.get('latency_ms', 0)
                        reranking_label = "Local Reranking"
                        reranking_help = f"{results['reranker']} reranking"

                    llm_generation_ms = st.session_state.get(f'rag_llm_latency_{method_name}', 0)
                    total_rag_latency = global_retrieval_ms + local_reranking_ms + llm_generation_ms

                    # Calculate percentages
                    if total_rag_latency > 0:
                        retrieval_pct = (global_retrieval_ms / total_rag_latency) * 100
                        reranking_pct = (local_reranking_ms / total_rag_latency) * 100
                        llm_pct = (llm_generation_ms / total_rag_latency) * 100
                    else:
                        retrieval_pct = reranking_pct = llm_pct = 0

                    # Create latency breakdown table
                    latency_data = []

                    # Row 1: Global Retrieval
                    latency_data.append({
                        "Metric": "1️⃣ Global Retrieval",
                        "Time (ms)": f"{global_retrieval_ms:.1f}",
                        "%": f"{retrieval_pct:.1f}%"
                    })

                    # Row 2: Local Reranking
                    latency_data.append({
                        "Metric": "2️⃣ Local Reranking",
                        "Time (ms)": f"{local_reranking_ms:.1f}",
                        "%": f"{reranking_pct:.1f}%"
                    })

                    # Row 3: LLM Generation
                    latency_data.append({
                        "Metric": "3️⃣ LLM Generation",
                        "Time (ms)": f"{llm_generation_ms:.1f}",
                        "%": f"{llm_pct:.1f}%"
                    })

                    # Row 4: Total
                    latency_data.append({
                        "Metric": "🏁 Total RAG Time",
                        "Time (ms)": f"{total_rag_latency:.1f}",
                        "%": "100.0%"
                    })

                    # Display as dataframe
                    latency_df = pd.DataFrame(latency_data)
                    st.dataframe(latency_df, use_container_width=True, hide_index=True)

                    # Show percentage breakdown and performance insights
                    if total_rag_latency > 0:

                        # Add performance insights for Real RAG Simulation mode
                        if is_rag_simulation:
                            st.markdown("---")
                            st.markdown("**💡 Performance Insights**")

                            # Calculate throughput
                            total_latency_sec = total_rag_latency / 1000
                            queries_per_sec = 1.0 / total_latency_sec if total_latency_sec > 0 else 0

                            col_i1, col_i2, col_i3 = st.columns(3)

                            with col_i1:
                                st.metric(
                                    "Throughput",
                                    f"{queries_per_sec:.2f} q/s",
                                    help="Queries per second for this RAG pipeline"
                                )

                            with col_i2:
                                # Estimate cost per 1M queries (assuming time-based cost)
                                time_per_million = total_latency_sec * 1_000_000 / 3600  # hours
                                st.metric(
                                    "Time/1M Queries",
                                    f"{time_per_million:.1f} hrs",
                                    help="Estimated time to process 1 million queries"
                                )

                            with col_i3:
                                # Show scalability indicator - use percentages calculated above
                                if llm_pct > 80:
                                    bottleneck = "LLM (GPU scaling recommended)"
                                elif retrieval_pct > 50:
                                    bottleneck = "Retrieval (index optimization)"
                                elif reranking_pct > 50:
                                    bottleneck = "Reranking (consider v2o)"
                                else:
                                    bottleneck = "Balanced ✅"

                                st.metric(
                                    "Bottleneck",
                                    bottleneck,
                                    help="Primary performance bottleneck"
                                )

                            # Comparison table if multiple evaluations exist
                            if f'rag_comparison_data' not in st.session_state:
                                st.session_state['rag_comparison_data'] = []

                            # Add button to save this run for comparison
                            if st.button("💾 Save for Comparison", key=f"save_comparison_{method_name}"):
                                comparison_entry = {
                                    'method': method_name,
                                    'reranker': results['reranker'],
                                    'docs': num_docs,
                                    'top_k': current_top_k,
                                    'retrieval_ms': global_retrieval_ms,
                                    'reranking_ms': local_reranking_ms,
                                    'llm_ms': llm_generation_ms,
                                    'total_ms': total_rag_latency,
                                    'llm_model': rag_llm_model
                                }
                                st.session_state['rag_comparison_data'].append(comparison_entry)
                                st.success(f"✅ Saved: {method_name} with {num_docs:,} docs")

                            # Show comparison table if we have saved data
                            if len(st.session_state.get('rag_comparison_data', [])) > 0:
                                st.markdown("**📊 Saved Comparisons**")

                                comparison_df = pd.DataFrame(st.session_state['rag_comparison_data'])

                                # Format the dataframe for display
                                display_df = comparison_df.copy()
                                display_df['Docs'] = display_df['docs'].apply(lambda x: f"{x:,}")
                                display_df['Retrieval'] = display_df['retrieval_ms'].apply(lambda x: f"{x:.1f}ms")
                                display_df['Reranking'] = display_df['reranking_ms'].apply(lambda x: f"{x:.1f}ms" if x > 0 else "—")
                                display_df['LLM'] = display_df['llm_ms'].apply(lambda x: f"{x:.1f}ms")
                                display_df['Total'] = display_df['total_ms'].apply(lambda x: f"{x:.1f}ms")

                                # Select columns to display
                                display_df = display_df[['method', 'reranker', 'Docs', 'top_k', 'Retrieval', 'Reranking', 'LLM', 'Total', 'llm_model']]
                                display_df.columns = ['Method', 'Reranker', 'Corpus Size', 'Top-K', 'Retrieval', 'Reranking', 'LLM', 'Total', 'LLM Model']

                                st.dataframe(display_df, use_container_width=True, hide_index=True)

                                # Add clear button
                                if st.button("🗑️ Clear Comparisons", key="clear_comparisons"):
                                    st.session_state['rag_comparison_data'] = []
                                    st.rerun()
            else:
                st.info(f"Click 'Generate Answer' to create a RAG response using top-{current_top_k} documents from {method_name}.")

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
