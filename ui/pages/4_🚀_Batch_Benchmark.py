"""
Batch Benchmark Execution Page

Automatically runs benchmarks on all datasets with all rerankers.
Designed for comprehensive weekend execution.
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

from utils.models import (
    load_all_models,
    run_reranker,
    get_model_info,
    RerankerModelError
)
from utils.metrics import calculate_all_metrics
from utils.data_loader import load_mteb_dataset

st.set_page_config(page_title="Batch Benchmark", layout="wide", page_icon="🚀")
st.header("🚀 Batch Benchmark Execution")

# ============================================================================
# Configuration
# ============================================================================

# Import datasets and reranker mappings from config
from config import DATASETS, RERANKER_SHORT_MAP, RE_RANKERS, MANISCOPE_VERSIONS, DEFAULT_MANISCOPE_VERSION

# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.markdown("### Batch Configuration")

    # Dataset selection
    st.markdown("#### Datasets")
    dataset_selection = st.multiselect(
        "Select Datasets",
        options=[d["name"] for d in DATASETS],
        default=[d["name"] for d in DATASETS],
        help="Choose which datasets to benchmark"
    )

    # ReRanker selection
    st.markdown("#### ReRankers")
    reranker_selection = st.multiselect(
        "Select ReRanker",
        options=RE_RANKERS,
        default=RE_RANKERS,
        help="Choose which reranker to benchmark"
    )

    st.markdown("---")

    # Maniscope configuration
    st.markdown("#### Maniscope Config")
    maniscope_k = st.number_input(
        "k (neighbors)",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        help="Number of nearest neighbors for manifold graph"
    )

    maniscope_alpha = st.number_input(
        "α (hybrid weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="0=pure geodesic, 1=pure cosine"
    )

    # ManiscopeEngine version selection
    version_options = list(MANISCOPE_VERSIONS.keys())
    version_labels = {v: MANISCOPE_VERSIONS[v]["name"] for v in version_options}

    default_index = version_options.index(DEFAULT_MANISCOPE_VERSION) if DEFAULT_MANISCOPE_VERSION in version_options else version_options.index('v2o')

    maniscope_version = st.selectbox(
        "ManiscopeEngine Version",
        options=version_options,
        format_func=lambda x: version_labels[x],
        index=default_index,
        help="Select optimization level for Maniscope"
    )

    # Display version info
    version_info = MANISCOPE_VERSIONS[maniscope_version]
    st.info(f"""
    **{version_info['name']}**
    - Latency: {version_info['latency']}
    - Speedup: {version_info['speedup']}
    - Best for: {version_info['best_for']}
    """)

    st.markdown("---")

    # LLM configuration
    if "LLM-Reranker" in reranker_selection:
        st.markdown("#### LLM Config")
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
                "ReRanker",
                ["llama3.1:latest", "deepseek-r1:7b", "qwen2.5:latest"],
                help="Ollama LLM model to use"
            )
        else:
            import os
            default_api_key = os.getenv("OPENROUTER_API_KEY", "")

            llm_base_url = "https://openrouter.ai/api/v1"
            llm_api_key = st.text_input(
                "API Key",
                value=default_api_key,
                type="password",
                help="Get your key at https://openrouter.ai"
            )
            llm_model = st.selectbox(
                "ReRanker",
                ["google/gemini-2.0-flash-lite-001", "anthropic/claude-3.5-haiku"],
                help="OpenRouter LLM model to use"
            )

        # Save to session state
        st.session_state['llm_base_url'] = llm_base_url
        st.session_state['llm_api_key'] = llm_api_key
        st.session_state['llm_model'] = llm_model

# ============================================================================
# Main Panel
# ============================================================================

# Calculate total work
selected_datasets = [d for d in DATASETS if d["name"] in dataset_selection]
total_queries = sum(d["queries"] for d in selected_datasets)
total_runs = len(selected_datasets) * len(reranker_selection)

# Display configuration summary
st.markdown("### Configuration Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Datasets", len(selected_datasets))
with col2:
    st.metric("Models", len(reranker_selection))
with col3:
    st.metric("Total Queries", total_queries)
with col4:
    st.metric("Total Runs", total_runs)

# Show selected datasets
if selected_datasets:
    st.markdown("#### Selected Datasets")
    dataset_table = []
    for d in selected_datasets:
        priority_label = "arXiv/ICML" if d["priority"] == 1 else "NeurIPS"
        dataset_table.append({
            "Dataset": d["name"],
            "Queries": d["queries"],
            "Priority": priority_label,
            "Description": d["description"]
        })
    st.dataframe(dataset_table, use_container_width=True, hide_index=True)

# Show selected rerankers
if reranker_selection:
    st.markdown("#### Selected Models")
    reranker_info_list = []
    for reranker_name in reranker_selection:
        info = get_model_info(reranker_name)
        reranker_info_list.append({
            "ReRanker": reranker_name,
            "Architecture": info["architecture"],
            "Description": info["description"]
        })
    st.dataframe(reranker_info_list, use_container_width=True, hide_index=True)

st.markdown("---")

# Estimated time
if total_queries > 0:
    # Rough estimates (queries per minute for 4 rerankers)
    est_minutes = total_queries * 0.5  # ~30 seconds per query with 4 rerankers
    est_hours = est_minutes / 60
    st.info(f"⏱️ Estimated time: {est_hours:.1f} hours ({est_minutes:.0f} minutes)")

# Run button
run_batch = st.button(
    "🚀 Run Batch Benchmark",
    type="primary",
    disabled=(len(selected_datasets) == 0 or len(reranker_selection) == 0),
    use_container_width=True
)

if not selected_datasets or not reranker_selection:
    st.warning("⚠️ Please select at least one dataset and one reranker in the sidebar")

# ============================================================================
# Batch Execution
# ============================================================================

if run_batch:
    # Store config in session
    st.session_state['maniscope_k'] = maniscope_k
    st.session_state['maniscope_alpha'] = maniscope_alpha
    st.session_state['maniscope_version'] = maniscope_version

    # Track results
    batch_results = []
    batch_start_time = time.time()

    # Overall progress
    overall_progress = st.progress(0)
    overall_status = st.empty()

    # Results summary placeholder
    results_summary = st.empty()

    # Load rerankers once
    try:
        with st.spinner(f"Loading {len(reranker_selection)} rerankers..."):
            rerankers = load_all_models(reranker_selection)

        st.success(f"✅ Loaded {len(rerankers)} rerankers successfully")

        # Iterate through datasets
        for dataset_idx, dataset_config in enumerate(selected_datasets):
            dataset_name = dataset_config["name"]
            dataset_file = dataset_config["file"]
            dataset_queries = dataset_config["queries"]

            overall_status.markdown(f"### 📊 Dataset {dataset_idx+1}/{len(selected_datasets)}: **{dataset_name}** ({dataset_queries} queries)")

            # Load dataset
            data_dir = Path(__file__).parent.parent.parent / "data"
            dataset_path = data_dir / dataset_file

            if not dataset_path.exists():
                st.error(f"❌ Dataset not found: {dataset_path}")
                continue

            with st.spinner(f"Loading {dataset_name}..."):
                with open(dataset_path, 'r') as f:
                    dataset = load_mteb_dataset(f, validate=True)

            st.info(f"📁 Loaded {len(dataset)} queries from {dataset_name}")

            # Run benchmark for this dataset
            results = []

            # Progress tracking for this dataset
            dataset_progress = st.progress(0)
            query_status = st.empty()
            metrics_display = st.empty()

            # Track timing
            dataset_start_time = time.time()
            model_times = {name: [] for name in rerankers.keys()}

            # Process queries
            for i, item in enumerate(dataset):
                query_status.text(f"Query {i+1}/{len(dataset)}: {item['query'][:60]}...")

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
                            "rankings": list(range(len(item['docs']))),
                            "latency_ms": 0.0,
                            "metrics": {metric: 0.0 for metric in ["MRR", "NDCG@3", "NDCG@10", "P@1", "P@3", "P@10", "R@10", "MAP"]}
                        }

                results.append(entry)

                # Update progress
                dataset_progress.progress((i + 1) / len(dataset))

                # Show intermediate metrics every 10 queries
                if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                    with metrics_display.container():
                        st.markdown(f"**Progress: {i+1}/{len(dataset)} queries**")

                        # Calculate average metrics so far
                        avg_metrics = {}
                        for reranker_name in rerankers.keys():
                            model_results = [r["rerankers"].get(reranker_name, {}) for r in results]
                            if model_results:
                                avg_metrics[reranker_name] = {
                                    "MRR": np.mean([m.get("metrics", {}).get("MRR", 0) for m in model_results]),
                                    "NDCG@3": np.mean([m.get("metrics", {}).get("NDCG@3", 0) for m in model_results]),
                                    "Latency": np.mean(model_times.get(reranker_name, [0]))
                                }

                        cols = st.columns(len(rerankers))
                        for idx, (reranker_name, metrics) in enumerate(avg_metrics.items()):
                            with cols[idx]:
                                st.metric(reranker_name, f"MRR: {metrics['MRR']:.3f}")
                                st.caption(f"NDCG@3: {metrics['NDCG@3']:.3f}")
                                st.caption(f"Latency: {metrics['Latency']:.1f}ms")

            # Dataset complete
            dataset_time = time.time() - dataset_start_time

            dataset_progress.empty()
            query_status.empty()
            metrics_display.empty()

            st.success(f"✅ {dataset_name} complete in {dataset_time:.1f}s!")

            # Save results for this dataset (IMPORTANT: saves after EACH dataset)
            st.markdown(f"### 💾 Saving Results for {dataset_name}...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(__file__).parent.parent.parent / "output"
            output_dir.mkdir(exist_ok=True)

            # Build filename using mappings from config
            reranker_shorts = sorted([RERANKER_SHORT_MAP.get(m, m.lower()[:4]) for m in reranker_selection])
            model_list_str = '-'.join(reranker_shorts)

            k_val = maniscope_k if 'Maniscope' in reranker_selection else 0
            version_suffix = f"-{maniscope_version}" if 'Maniscope' in reranker_selection else ""
            k_prefix = f"k{k_val}{version_suffix}" if 'Maniscope' in reranker_selection else "k0"

            dataset_short = dataset_config["short"]
            results_filename = output_dir / f"{k_prefix}-{dataset_short}_{dataset_queries}-{model_list_str}_{timestamp}.json"

            results_data = {
                "metadata": {
                    "dataset_name": dataset_name,
                    "dataset_short": dataset_short,
                    "dataset_path": str(dataset_path),
                    "dataset_source": "batch_benchmark",
                    "num_queries": len(results),
                    "rerankers": reranker_selection,
                    "reranker_shorts": reranker_shorts,
                    "maniscope_k": maniscope_k if 'Maniscope' in reranker_selection else None,
                    "maniscope_alpha": maniscope_alpha if 'Maniscope' in reranker_selection else None,
                    "maniscope_version": maniscope_version if 'Maniscope' in reranker_selection else None,
                    "timestamp": timestamp,
                    "total_time_seconds": dataset_time,
                    "filename": str(results_filename.name)
                },
                "results": results
            }

            try:
                # Save to file
                with open(results_filename, 'w') as f:
                    json.dump(results_data, f, indent=2)

                # Verify file was created
                if results_filename.exists():
                    file_size = results_filename.stat().st_size / 1024  # KB
                    st.success(f"✅ **File saved successfully!**")
                    st.info(f"""
                    📁 **File:** `{results_filename.name}`
                    📂 **Path:** `{results_filename.resolve()}`
                    💾 **Size:** {file_size:.1f} KB
                    📊 **Queries:** {len(results)}
                    🤖 **Models:** {len(reranker_selection)}
                    """)
                else:
                    st.error(f"❌ File save verification failed: {results_filename}")

            except Exception as e:
                st.error(f"❌ Failed to save results for {dataset_name}: {str(e)}")
                st.exception(e)

            # Track for summary
            batch_results.append({
                "dataset": dataset_name,
                "queries": len(results),
                "time": dataset_time,
                "file": str(results_filename.resolve()),  # Full path
                "metrics": {
                    reranker_name: {
                        "MRR": np.mean([r["rerankers"].get(reranker_name, {}).get("metrics", {}).get("MRR", 0) for r in results]),
                        "NDCG@3": np.mean([r["rerankers"].get(reranker_name, {}).get("metrics", {}).get("NDCG@3", 0) for r in results]),
                        "Latency": np.mean(model_times.get(reranker_name, [0]))
                    }
                    for reranker_name in rerankers.keys()
                }
            })

            st.markdown("---")  # Visual separator between datasets

            # Update overall progress
            overall_progress.progress((dataset_idx + 1) / len(selected_datasets))

        # Batch complete
        batch_time = time.time() - batch_start_time

        overall_progress.empty()
        overall_status.empty()

        st.markdown("---")
        st.success(f"🎉 Batch benchmark complete! Total time: {batch_time/60:.1f} minutes ({batch_time:.1f}s)")

        # Summary table
        st.markdown("### 📊 Batch Results Summary")

        summary_rows = []
        for result in batch_results:
            for reranker_name, metrics in result["metrics"].items():
                summary_rows.append({
                    "Dataset": result["dataset"],
                    "Queries": result["queries"],
                    "ReRanker": reranker_name,
                    "MRR": f"{metrics['MRR']:.4f}",
                    "NDCG@3": f"{metrics['NDCG@3']:.4f}",
                    "Latency (ms)": f"{metrics['Latency']:.1f}",
                    "Time (min)": f"{result['time']/60:.1f}"
                })

        import pandas as pd
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Show output files with verification
        st.markdown("### 💾 Saved Output Files")
        st.markdown(f"**Total files saved:** {len(batch_results)}")

        for idx, result in enumerate(batch_results, 1):
            file_path = Path(result["file"])
            if file_path.exists():
                file_size = file_path.stat().st_size / 1024  # KB
                st.success(f"✅ **File {idx}/{len(batch_results)}: {result['dataset']}** ({result['queries']} queries)")
                st.code(str(file_path), language=None)
                st.caption(f"Size: {file_size:.1f} KB | Models: {len(reranker_selection)} | Time: {result['time']:.1f}s")
            else:
                st.error(f"❌ File {idx}: {result['dataset']} - File not found!")
                st.code(result["file"], language=None)

        st.markdown("---")
        st.info("👉 Go to the **Analytics** page to view detailed visualizations of each result file")

        # Quick analysis command
        st.markdown("### 📊 Quick Analysis")
        st.markdown("Run this command to analyze all results:")
        output_dir = Path(__file__).parent.parent.parent / "output"
        today_date = datetime.now().strftime("%Y%m%d")
        analysis_cmd = f"cd {output_dir.parent} && python3 analyze_results.py output/k*-*_{today_date}_*.json"
        st.code(analysis_cmd, language="bash")

    except RerankerModelError as e:
        st.error(f"❌ ReRanker loading failed: {str(e)}")
        st.info("💡 Make sure FlagEmbedding is installed: `pip install -U FlagEmbedding`")
    except Exception as e:
        st.error(f"❌ Batch benchmark failed: {str(e)}")
        st.exception(e)

# ============================================================================
# Information Panel (when not running)
# ============================================================================

if not run_batch:
    st.markdown("---")
    st.markdown("### ℹ️ About Batch Benchmarking")

    st.markdown("""
    This page automates comprehensive benchmark execution across multiple datasets and models.

    **Purpose:**
    - Run full benchmarks on all datasets with all rerankers automatically
    - Designed for weekend/overnight execution
    - Generates complete results for paper submission

    **Process:**
    1. Select datasets and rerankers in sidebar
    2. Configure Maniscope and LLM parameters
    3. Click "Run Batch Benchmark"
    4. Wait for completion (estimated time shown above)
    5. Results auto-saved to `/output/` directory

    **Use Cases:**
    - **arXiv/ICML:** Run Priority 1 datasets (SciFact, MS MARCO, TREC-COVID)
    - **NeurIPS:** Run all datasets including Priority 2 (ArguAna, FiQA)
    - **Validation:** Run all datasets with all rerankers for comprehensive evaluation

    **Tips:**
    - Start with a single dataset to verify everything works
    - Run overnight for full benchmark suite
    - Results can be analyzed in the Analytics page
    - Use the analysis script for detailed comparisons
    """)

    with st.expander("📋 Expected Output Files"):
        st.markdown("""
        Results saved with format: `<k-val>-<version>-<dataset>_<queries>-<rerankers>_<timestamp>.json`

        **Example files:**
        ```
        k5-v2o-sci_100-bge-jina-llm-mani_20260120_140000.json
        k5-v2o-marco_200-bge-jina-llm-mani_20260120_150000.json
        k5-v3-covid_50-bge-jina-llm-mani_20260120_160000.json
        k5-v2o-arg_100-bge-jina-llm-mani_20260120_170000.json
        k5-v2o-fiqa_100-bge-jina-llm-mani_20260120_180000.json
        ```

        **Note:** Version suffix (e.g., `-v2o`, `-v3`) is included when Maniscope is selected.
        """)
