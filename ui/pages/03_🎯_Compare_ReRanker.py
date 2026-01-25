"""
Benchmark Execution Page

Runs rerankers on the loaded dataset and saves results.
"""

import streamlit as st
import json
import time
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATASETS,
    RE_RANKERS,
    RERANKER_SHORT_MAP,
    DEFAULT_MANISCOPE_K,
    DEFAULT_MANISCOPE_ALPHA,
    OPENROUTER_MODELS,
    OLLAMA_MODELS,
    DEFAULT_OPENROUTER_URL,
    DEFAULT_OLLAMA_URL,
    COLORS,
    PAGE_ICON,
    PAGE_LAYOUT,
    OUTPUT_DIRS,
    ensure_output_dirs
)

from utils.models import (
    load_all_models,
    run_reranker,
    get_model_info,
    RerankerModelError
)
from utils.metrics import calculate_all_metrics

st.set_page_config(page_title="Run Benchmark", layout="wide", page_icon="🎯")

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================
with st.sidebar:
    # Dataset selection
    # st.markdown("#### Dataset Selection")

    # Get available datasets from config
    dataset_options = {d['name']: d for d in DATASETS}
    dataset_names = sorted(list(dataset_options.keys()))

    # Determine current dataset index
    current_dataset_idx = 0
    if 'dataset_name' in st.session_state and st.session_state['dataset_name']:
        # Try to match by filename
        current_name = st.session_state['dataset_name']
        for idx, name in enumerate(dataset_names):
            if dataset_options[name]['file'] in current_name or name in current_name:
                current_dataset_idx = idx
                break

    selected_dataset_name = st.selectbox(
        "Select Dataset",
        options=dataset_names,
        index=current_dataset_idx,
        help="Choose a dataset to benchmark on"
    )

    # Load dataset if not loaded or if changed
    selected_config = dataset_options[selected_dataset_name]
    dataset_file = selected_config['file']
    dataset_path = Path(__file__).parent.parent.parent / "data" / dataset_file

    if ('dataset' not in st.session_state or
        st.session_state['dataset'] is None or
        st.session_state.get('dataset_name') != dataset_file):

        if dataset_path.exists():
            try:
                import json
                from utils.data_loader import load_mteb_dataset

                with open(dataset_path, 'r') as f:
                    data = load_mteb_dataset(f, validate=True)
                    st.session_state['dataset'] = data
                    st.session_state['dataset_name'] = dataset_file
                    st.session_state['dataset_path'] = str(dataset_path.resolve())
                    st.success(f"✅ Loaded {selected_dataset_name} ({len(data)} queries)")
            except Exception as e:
                st.error(f"❌ Error loading {selected_dataset_name}: {e}")
                st.stop()
        else:
            st.error(f"❌ Dataset file not found: {dataset_file}")
            st.info("💡 Go to Data Manager to load a custom dataset, or check that the dataset file exists in the data/ directory.")
            st.stop()

    # Display dataset info
    if 'dataset' in st.session_state and st.session_state['dataset']:
        dataset = st.session_state['dataset']
    else:
        st.error("❌ No dataset loaded.")
        st.stop()

    dataset_name = st.session_state['dataset_name']

    # st.markdown("---")

    # ReRanker selection
    # st.markdown("#### ReRankers")

    # reranker_selection = []
    # for reranker_name in RE_RANKERS:
    #     info = get_model_info(reranker_name)
    #     is_selected = st.checkbox(
    #         reranker_name,
    #         value=(reranker_name in ["Maniscope","HNSW", "BGE-M3", "Jina Reranker v2", ]),  # Only Maniscope by default
    #         help=f"{info['architecture']}\n\n{info['description']}"
    #     )
    #     if is_selected:
    #         reranker_selection.append(reranker_name)

    default_rerankers = ["Maniscope","HNSW", "Jina Reranker v2", "BGE-M3"]
    reranker_selection = st.multiselect(
        "Select ReRanker",
        options=RE_RANKERS,
        default=default_rerankers,
        help="Choose which reranker to benchmark"
    )            

    # Maniscope configuration (if selected)
    if "Maniscope" in reranker_selection:
        st.markdown("#### Maniscope Settings")

        c__1, c__2 = st.columns(2)
        with c__1:
            maniscope_k = st.number_input(
                "k (neighbors)",
                min_value=3,
                max_value=20,
                value=DEFAULT_MANISCOPE_K,
                step=1,
                help="Number of nearest neighbors for manifold graph"
            )
            st.session_state['maniscope_k'] = maniscope_k

        with c__2:
            maniscope_alpha = st.number_input(
                "α (hybrid weight)",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_MANISCOPE_ALPHA,
                step=0.1,
                help="0=pure geodesic, 1=pure cosine"
            )
            st.session_state['maniscope_alpha'] = maniscope_alpha


    if "LLM-Reranker" in reranker_selection:
        st.markdown("#### LLM Model Config")
        llm_provider = st.radio(
            "Provider",
            ["OpenRouter (Cloud)", "Ollama (Local)"],
            help="Choose between cloud OpenRouter or local Ollama"
        )

        if llm_provider == "Ollama (Local)":
            llm_base_url = st.text_input(
                "Base URL",
                value="http://localhost:11434/v1",
                help="Ollama API endpoint"
            )
            llm_api_key = "ollama"
            llm_model = st.selectbox(
                "LLM Model",
                ["llama3.1:latest", "llama3.2:latest", "mistral:latest", "deepseek-r1:7b", "qwen2.5:latest"],
                help="Ollama LLM model to use"
            )
        else:
            # OpenRouter - auto-load from environment variable
            import os
            default_api_key = os.getenv("OPENROUTER_API_KEY", "")

            llm_base_url = "https://openrouter.ai/api/v1"
            llm_api_key = st.text_input(
                "OpenRouter API Key",
                value=default_api_key,
                type="password",
                help="Get your key at https://openrouter.ai (auto-loaded from OPENROUTER_API_KEY env var)"
            )
            llm_model = st.selectbox(
                "LLM Model",
                ["google/gemini-2.0-flash-lite-001", "anthropic/claude-3.5-haiku"],
                help="OpenRouter LLM model to use"
            )

        # Save to session state
        st.session_state['llm_base_url'] = llm_base_url
        st.session_state['llm_api_key'] = llm_api_key
        st.session_state['llm_model'] = llm_model

        # st.markdown("---")

    # Execution Settings
    st.markdown("#### Execution Settings")

    c_1, c_2 = st.columns(2)
    with c_1:
        max_queries = st.number_input(
            "Max Queries",
            min_value=1,
            max_value=len(dataset),
            value=min(10, len(dataset)),
            help="Limit number of queries for quick tests"
        )

    with c_2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=len(dataset),
            value=min(5, len(dataset)),
            help="Process queries in batches for progress updates"
        )


    c_b1, c_b2 = st.columns([5,2])
    with c_b1:
        run_button = st.button(
            "Start Benchmark",
            type="primary",
            disabled=len(reranker_selection) == 0,
            use_container_width=True
        )

    with c_b2:
        clear_button = st.button(
            "Clear",
            use_container_width=True
        )

    # st.divider()
    st.markdown("---")
    st.markdown(f"**Dataset:** `{dataset_path}`  ({len(dataset)} queries)")
    st.info(f"📊 **{selected_dataset_name}**: {selected_config['description']}")


    if clear_button:
        if 'results' in st.session_state:
            del st.session_state['results']
        if 'results_file' in st.session_state:
            del st.session_state['results_file']
        st.success("Results cleared!")
        st.rerun()

    if not reranker_selection:
        st.warning("⚠️ Select at least one reranker above")

# ============================================================================
# MAIN PANEL - Results Only
# ============================================================================
st.header("🎯 Benchmark Results")

# Run benchmark
if run_button:
    try:
        # Load rerankers
        with st.spinner(f"Loading {len(reranker_selection)} rerankers..."):
            rerankers = load_all_models(reranker_selection)

        st.success(f"✅ Loaded {len(rerankers)} rerankers successfully")

        # Show which rerankers loaded
        for reranker_name in rerankers.keys():
            info = get_model_info(reranker_name)
            st.caption(f"  • {reranker_name}: {info['architecture']}")

        # Initialize results
        results = []
        queries_to_process = dataset[:max_queries]

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        # Track timing
        start_time = time.time()
        model_times = {name: [] for name in rerankers.keys()}

        # Process queries
        for i, item in enumerate(queries_to_process):
            status_text.text(f"Processing query {i+1}/{len(queries_to_process)}: {item['query'][:50]}...")

            entry = {
                "query_id": item.get("query_id", str(i)),
                "query": item["query"],
                "relevance_map": item["relevance_map"],
                "rerankers": {}
            }

            # Run each reranker
            for reranker_name, reranker in rerankers.items():
                try:
                    # Time the reranking
                    model_start = time.perf_counter()
                    scores = run_reranker(reranker, item['query'], item['docs'])
                    model_latency = (time.perf_counter() - model_start) * 1000  # ms

                    model_times[reranker_name].append(model_latency)

                    # Get rankings
                    rankings = np.argsort(scores)[::-1].tolist()

                    # Calculate metrics
                    metrics = calculate_all_metrics(rankings, item['relevance_map'])

                    entry["rerankers"][reranker_name] = {
                        "scores": scores.tolist(),
                        "rankings": rankings,
                        "latency_ms": model_latency,
                        "metrics": metrics
                    }

                except Exception as e:
                    st.warning(f"⚠️ {reranker_name} failed on query {i+1}: {str(e)}")
                    entry["rerankers"][reranker_name] = {
                        "error": str(e),
                        "rankings": list(range(len(item['docs']))),  # Default ranking
                        "latency_ms": 0.0,
                        "metrics": {metric: 0.0 for metric in ["MRR", "NDCG@3", "NDCG@10", "P@1", "P@3", "P@10", "R@10", "MAP"]}
                    }

            results.append(entry)

            # Update progress
            progress_bar.progress((i + 1) / len(queries_to_process))

            # Show intermediate metrics every batch
            if (i + 1) % batch_size == 0 or (i + 1) == len(queries_to_process):
                with metrics_placeholder.container():
                    st.markdown(f"**Progress: {i+1}/{len(queries_to_process)} queries**")

                    # Calculate average metrics so far
                    avg_metrics = {}
                    for reranker_name in rerankers.keys():
                        model_results = [r["rerankers"].get(reranker_name, {}) for r in results]
                        if model_results:
                            avg_metrics[reranker_name] = {
                                "MRR": np.mean([m.get("metrics", {}).get("MRR", 0) for m in model_results]),
                                "NDCG@3": np.mean([m.get("metrics", {}).get("NDCG@3", 0) for m in model_results]),
                                "Avg Latency": np.mean(model_times.get(reranker_name, [0]))
                            }

                    cols = st.columns(len(rerankers))
                    for idx, (reranker_name, metrics) in enumerate(avg_metrics.items()):
                        with cols[idx]:
                            st.metric(reranker_name, f"MRR: {metrics['MRR']:.3f}")
                            st.caption(f"NDCG@3: {metrics['NDCG@3']:.3f}")
                            st.caption(f"Latency: {metrics['Avg Latency']:.1f}ms")

        # Benchmark complete
        total_time = time.time() - start_time

        progress_bar.empty()
        status_text.empty()
        metrics_placeholder.empty()

        st.success(f"✅ Benchmark complete in {total_time:.1f}s!")

        # Save results to output folder with descriptive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_output_dirs()  # Ensure all output directories exist
        output_dir = OUTPUT_DIRS["benchmark"]

        # Extract dataset short name from config
        dataset_short = 'unknown'
        dataset_lower = dataset_name.lower().replace('dataset-', '').replace('.json', '')
        for dataset_config in DATASETS:
            if dataset_config['short'] in dataset_lower or dataset_config['file'] == dataset_name:
                dataset_short = dataset_config['short']
                break

        # Map reranker names to short names from config (sorted alphabetically)
        reranker_shorts = sorted([RERANKER_SHORT_MAP.get(m, m.lower()[:4]) for m in reranker_selection])
        reranker_list_str = '-'.join(reranker_shorts)

        # Get k value (if Maniscope is used)
        k_val = st.session_state.get('maniscope_k', 5) if 'Maniscope' in reranker_selection else 0
        k_prefix = f"k{k_val}" if 'Maniscope' in reranker_selection else "k0"

        # Build filename: <k-val>-<dataset>_<max_queries>-<rerankers>_<timestamp>.json
        results_filename = output_dir / f"{k_prefix}-{dataset_short}_{max_queries}-{reranker_list_str}_{timestamp}.json"
        results_full_path = str(results_filename.resolve())

        results_data = {
            "metadata": {
                "dataset_name": dataset_name,
                "dataset_short": dataset_short,
                "dataset_path": str(dataset_path),
                "dataset_source": st.session_state.get('dataset_source', 'unknown'),
                "num_queries": len(results),
                "max_queries_requested": max_queries,
                "rerankers": reranker_selection,
                "reranker_shorts": reranker_shorts,
                "maniscope_k": st.session_state.get('maniscope_k', None),
                "maniscope_alpha": st.session_state.get('maniscope_alpha', None),
                "timestamp": timestamp,
                "total_time_seconds": total_time,
                "filename": str(results_filename.name)
            },
            "results": results
        }

        with open(results_filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        st.session_state['results'] = results
        st.session_state['results_metadata'] = results_data['metadata']
        st.session_state['results_file'] = results_full_path

        st.info(f"💾 Results saved to:\n\n`{results_full_path}`")

        # Final summary
        st.markdown("### Benchmark Summary")

        summary_cols = st.columns(len(rerankers))
        for idx, reranker_name in enumerate(rerankers.keys()):
            with summary_cols[idx]:
                model_results = [r["rerankers"].get(reranker_name, {}) for r in results if reranker_name in r["rerankers"]]

                if model_results:
                    avg_mrr = np.mean([m.get("metrics", {}).get("MRR", 0) for m in model_results])
                    avg_ndcg3 = np.mean([m.get("metrics", {}).get("NDCG@3", 0) for m in model_results])
                    avg_latency = np.mean(model_times.get(reranker_name, [0]))

                    st.markdown(f"**{reranker_name}**")
                    st.metric("MRR", f"{avg_mrr:.4f}")
                    st.metric("NDCG@3", f"{avg_ndcg3:.4f}")
                    st.metric("Avg Latency", f"{avg_latency:.1f} ms")

        # st.markdown("---")
        st.info("👉 Go to the Analytics page to view detailed results and visualizations")

    except RerankerModelError as e:
        st.error(f"❌ ReRanker loading failed: {str(e)}")
        st.info("💡 Make sure FlagEmbedding is installed: `pip install -U FlagEmbedding`")
    except Exception as e:
        st.error(f"❌ Benchmark failed: {str(e)}")
        st.exception(e)

# ============================================================================
# Show existing results
# ============================================================================
if 'results' in st.session_state and not run_button:
    metadata = st.session_state.get('results_metadata', {})
    results_file = st.session_state.get('results_file', 'N/A')

    st.info(f"📊 Results available from last run:\n\n`{results_file}`")

    col1, col2, col3, col4 = st.columns([8,3,3,3])
    with col1:
        st.metric("Dataset", metadata.get('dataset_name', 'N/A'))
    with col2:
        st.metric("Queries", metadata.get('num_queries', 0))
    with col3:
        st.metric("ReRankers", len(metadata.get('rerankers', [])))
    with col4:
        st.metric("Runtime", f"{metadata.get('total_time_seconds', 0):.1f}s")

    st.caption("Go to Analytics page to view detailed results")
