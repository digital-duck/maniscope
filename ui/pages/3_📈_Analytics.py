"""
Analytics Dashboard

Comprehensive visualization and analysis of reranker benchmark results.

Includes:
- Metric overview and comparisons
- Per-query analysis
- Latency analysis
- Data export functionality
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import io

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import (
    plot_metric_comparison,
    plot_all_metrics_comparison,
    plot_latency_distribution,
    plot_pareto_frontier,
    plot_per_query_heatmap,
    plot_metric_over_queries,
    create_summary_table
)

st.set_page_config(page_title="Analytics", layout="wide", page_icon="📈")
st.header("📈 Analytics Dashboard")

# ============================================================================
# Sidebar: File Browser
# ============================================================================

with st.sidebar:
    st.markdown("### 📁 Load Results")

    # Scan output directory for result files
    output_dir = Path(__file__).parent.parent.parent / "output"

    if output_dir.exists():
        # Get all JSON files, sorted by modification time (newest first)
        result_files = sorted(
            output_dir.glob("k*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if result_files:
            # Create display names with metadata
            file_options = {}
            file_details = {}

            for file_path in result_files:
                try:
                    # Try to read metadata for display
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        meta = data.get('metadata', {})

                    # Create display name
                    dataset = meta.get('dataset_short', 'unknown')
                    queries = meta.get('num_queries', '?')

                    # Handle both 'models' and 'rerankers' fields (backward compatibility)
                    models_list = meta.get('models', meta.get('rerankers', []))
                    models = len(models_list)

                    timestamp = meta.get('timestamp', '')[:8]  # YYYYMMDD

                    # Format: "sci_100 (4 models) - 20260118"
                    display_name = f"{dataset}_{queries} ({models} models) - {timestamp}"

                    file_options[display_name] = str(file_path)
                    file_details[display_name] = {
                        'path': file_path,
                        'size': file_path.stat().st_size / 1024,  # KB
                        'meta': meta
                    }
                except:
                    # If can't read metadata, use filename
                    display_name = file_path.name
                    file_options[display_name] = str(file_path)
                    file_details[display_name] = {
                        'path': file_path,
                        'size': file_path.stat().st_size / 1024,
                        'meta': {}
                    }

            # Selectbox for file selection
            selected_display = st.selectbox(
                "Select Result File",
                options=list(file_options.keys()),
                help="Choose a result file from the output directory"
            )

            if selected_display:
                selected_file_path = file_details[selected_display]['path']
                file_size = file_details[selected_display]['size']

                # Show file info
                st.caption(f"📄 {selected_file_path.name}")
                st.caption(f"💾 Size: {file_size:.1f} KB")

                # Load button
                if st.button("📂 Load Result", type="primary", use_container_width=True):
                    try:
                        with open(selected_file_path, 'r') as f:
                            data = json.load(f)

                        # Load into session state
                        if "metadata" in data and "results" in data:
                            metadata = data['metadata']

                            # Normalize metadata: handle both 'models' and 'rerankers' for backward compatibility
                            if 'rerankers' in metadata and 'models' not in metadata:
                                metadata['models'] = metadata['rerankers']
                                metadata['model_shorts'] = metadata.get('reranker_shorts', [])

                            st.session_state['results'] = data['results']
                            st.session_state['results_metadata'] = metadata
                            st.session_state['results_file'] = str(selected_file_path)
                            st.success(f"✅ Loaded {len(data['results'])} results!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid result file format")
                    except Exception as e:
                        st.error(f"❌ Failed to load: {str(e)}")

            st.markdown("---")
        else:
            st.warning("⚠️ No result files found in output directory")
    else:
        st.warning("⚠️ Output directory not found")

    st.markdown("**Or upload a file:**")
    uploaded_file = st.file_uploader(
        "Upload Results JSON",
        type=['json'],
        help="Upload a results file from elsewhere",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)

            if "metadata" in data and "results" in data:
                metadata = data['metadata']

                # Normalize metadata: handle both 'models' and 'rerankers' for backward compatibility
                if 'rerankers' in metadata and 'models' not in metadata:
                    metadata['models'] = metadata['rerankers']
                    metadata['model_shorts'] = metadata.get('reranker_shorts', [])

                st.session_state['results'] = data['results']
                st.session_state['results_metadata'] = metadata
                st.session_state['results_file'] = f"uploaded: {uploaded_file.name}"
                st.success(f"✅ Loaded {len(data['results'])} results!")
                st.rerun()
            else:
                st.session_state['results'] = data
                st.session_state['results_metadata'] = {'dataset_name': uploaded_file.name}
                st.session_state['results_file'] = f"uploaded: {uploaded_file.name}"
                st.success(f"✅ Loaded {len(data)} results!")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to upload: {str(e)}")

# ============================================================================
# Check if Results Loaded
# ============================================================================

if 'results' not in st.session_state or st.session_state['results'] is None:
    st.info("ℹ️ No results loaded yet.")
    st.markdown("👈 **Use the sidebar** to:")
    st.markdown("1. Select a result file from the dropdown")
    st.markdown("2. Click 'Load Selected File'")
    st.markdown("3. Or upload a file from elsewhere")
    st.stop()

# Get results from session state
results = st.session_state['results']
metadata = st.session_state.get('results_metadata', {})
results_file = st.session_state.get('results_file', 'N/A')
dataset_path = metadata.get('dataset_path', metadata.get('dataset_name', 'N/A'))

# ============================================================================
# Sidebar: Current Results Info
# ============================================================================

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Current Results")
    st.success(f"✅ **Loaded**")

    # Extract filename from full path
    if results_file.startswith('uploaded:'):
        display_file = results_file
    else:
        display_file = Path(results_file).name if results_file != 'N/A' else results_file

    st.caption(f"📁 {display_file}")
    st.markdown(f"**Dataset:** {metadata.get('dataset_short', 'N/A')}")
    st.markdown(f"**Queries:** {len(results)}")
    num_models = len(metadata.get('models', []))
    st.markdown(f"**Models:** {num_models}")
    if num_models > 0:
        models_list = ', '.join(metadata.get('models', []))
        st.caption(f"   ({models_list})")
    txt_runtime = f"{metadata.get('total_time_seconds', 0):.1f}"
    st.markdown(f"**Runtime:** {txt_runtime}s")
    txt_timestamp = metadata.get('timestamp', 'N/A')
    st.markdown(f"**Timestamp:** {txt_timestamp}")

# ============================================================================
# Create Tabs
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Metric Overview",
    "🔍 Per-Query Analysis",
    "⚡ Latency Analysis",
    "📊 Detailed Results",
    "💾 Export Data"
])

# ============================================================================
# TAB 1: Metric Overview
# ============================================================================

with tab1:
    st.markdown("### Metric Overview")

    # Summary table
    st.markdown("#### Summary Statistics")
    summary_df = create_summary_table(results)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Download summary
    csv = summary_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Summary CSV",
        csv,
        "summary_stats.csv",
        "text/csv"
    )

    st.markdown("---")

    # All metrics comparison
    st.markdown("#### All Metrics Comparison")
    fig = plot_all_metrics_comparison(results)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Individual metric charts
    st.markdown("#### Individual Metrics")

    metrics_to_plot = ["MRR", "NDCG@3", "NDCG@10", "MAP", "P@3", "R@10"]

    col1, col2 = st.columns(2)
    for i, metric in enumerate(metrics_to_plot):
        with col1 if i % 2 == 0 else col2:
            fig = plot_metric_comparison(results, metric)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: Per-Query Analysis
# ============================================================================

with tab2:
    st.markdown("### Per-Query Analysis")

    st.markdown("""
    Analyze how each model performs on individual queries.
    Use this to identify queries where models disagree or struggle.
    """)

    # Metric selector
    metric_for_heatmap = st.selectbox(
        "Select Metric",
        ["MRR", "NDCG@1", "NDCG@3", "NDCG@10", "P@1", "P@3", "MAP"],
        index=2
    )

    # Number of queries to show
    num_results = len(results)
    max_queries_to_show = st.slider(
        "Number of Queries to Display",
        min_value=min(1, num_results),
        max_value=min(num_results, 100),
        value=min(min(50, num_results), max(min(10, num_results), 1)),
        step=max(1, min(10, num_results // 10))
    )

    st.markdown("---")

    # Heatmap
    st.markdown("#### Performance Heatmap")
    fig = plot_per_query_heatmap(results, metric_for_heatmap, max_queries=max_queries_to_show)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Metric over queries
    st.markdown("#### Metric Trend Across Queries")
    fig = plot_metric_over_queries(results, metric_for_heatmap)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Find queries with largest disagreements
    st.markdown("#### Queries with Largest Model Disagreements")

    disagreements = []
    for i, result in enumerate(results[:100]):  # Limit to first 100
        query_id = result.get('query_id', str(i))
        query_text = result.get('query', '')[:80]

        model_scores = []
        for model_name, model_data in result.get('models', {}).items():
            score = model_data.get('metrics', {}).get(metric_for_heatmap, 0)
            model_scores.append(score)

        if len(model_scores) > 1:
            score_std = np.std(model_scores)
            disagreements.append({
                'Query ID': query_id,
                'Query': query_text,
                'Score StdDev': score_std,
                'Min Score': min(model_scores),
                'Max Score': max(model_scores)
            })

    if disagreements:
        df_disagreements = pd.DataFrame(disagreements)
        df_disagreements = df_disagreements.sort_values('Score StdDev', ascending=False).head(10)

        st.dataframe(df_disagreements, use_container_width=True, hide_index=True)
        st.caption("Queries where models have the most different scores (high standard deviation)")

# ============================================================================
# TAB 3: Latency Analysis
# ============================================================================

with tab3:
    st.markdown("### Latency Analysis")

    # Latency distribution
    st.markdown("#### Latency Distribution")
    fig = plot_latency_distribution(results)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Pareto frontier
    st.markdown("#### Efficiency Frontier (Accuracy vs Latency)")

    metric_for_pareto = st.selectbox(
        "Select Accuracy Metric",
        ["NDCG@3", "NDCG@10", "MRR", "MAP"],
        key="pareto_metric"
    )

    fig = plot_pareto_frontier(results, metric_for_pareto)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Latency statistics table
    st.markdown("#### Latency Statistics")

    latency_stats = []
    model_names = list(results[0].get('models', {}).keys())

    for model_name in model_names:
        latencies = [
            r.get('models', {}).get(model_name, {}).get('latency_ms', 0)
            for r in results
        ]

        latencies = [l for l in latencies if l > 0]  # Filter out zeros

        if latencies:
            latency_stats.append({
                'Model': model_name,
                'Mean (ms)': np.mean(latencies),
                'Median (ms)': np.median(latencies),
                'Std Dev (ms)': np.std(latencies),
                'Min (ms)': min(latencies),
                'Max (ms)': max(latencies),
                'Queries/sec': 1000 / np.mean(latencies)
            })

    if latency_stats:
        df_latency = pd.DataFrame(latency_stats)
        for col in df_latency.columns:
            if col != 'Model':
                df_latency[col] = df_latency[col].round(2)

        st.dataframe(df_latency, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 4: Detailed Results
# ============================================================================

with tab4:
    st.markdown("### Detailed Results")

    # Create detailed dataframe
    detailed_rows = []

    for i, result in enumerate(results):
        query_id = result.get('query_id', str(i))
        query_text = result.get('query', '')

        for model_name, model_data in result.get('models', {}).items():
            metrics = model_data.get('metrics', {})
            latency = model_data.get('latency_ms', 0)

            row = {
                'Query ID': query_id,
                'Query': query_text[:50] + '...' if len(query_text) > 50 else query_text,
                'Model': model_name,
                'Latency (ms)': latency
            }

            # Add all metrics
            for metric_name, value in metrics.items():
                row[metric_name] = value

            detailed_rows.append(row)

    df_detailed = pd.DataFrame(detailed_rows)

    # Round numeric columns
    numeric_cols = df_detailed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'Latency' in col:
            df_detailed[col] = df_detailed[col].round(2)
        else:
            df_detailed[col] = df_detailed[col].round(4)

    # Filters
    st.markdown("#### Filters")
    col1, col2 = st.columns(2)

    with col1:
        model_filter = st.multiselect(
            "Filter by Model",
            options=df_detailed['Model'].unique().tolist(),
            default=df_detailed['Model'].unique().tolist()
        )

    with col2:
        search_query = st.text_input(
            "Search Query Text",
            placeholder="Enter text to search in queries..."
        )

    # Apply filters
    df_filtered = df_detailed[df_detailed['Model'].isin(model_filter)]

    if search_query:
        df_filtered = df_filtered[df_filtered['Query'].str.contains(search_query, case=False, na=False)]

    st.markdown(f"### Results ({len(df_filtered)} rows)")
    st.dataframe(df_filtered, use_container_width=True, height=600)

# ============================================================================
# TAB 5: Export Data
# ============================================================================

with tab5:
    st.markdown("### Export Data")

    st.markdown("""
    Export results in various formats for further analysis or publication.
    """)

    # Export options
    export_format = st.radio(
        "Export Format",
        ["Excel (Multi-sheet)", "CSV (Summary)", "CSV (Detailed)", "JSON (Raw)"],
        help="Choose format based on your needs"
    )

    if st.button("📥 Generate Export", type="primary"):
        try:
            if export_format == "Excel (Multi-sheet)":
                # Create Excel with multiple sheets
                output = io.BytesIO()

                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Summary sheet
                    summary_df = create_summary_table(results)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                    # Detailed sheet
                    df_detailed.to_excel(writer, sheet_name='Detailed', index=False)

                    # Metadata sheet
                    metadata_df = pd.DataFrame([metadata])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

                excel_data = output.getvalue()

                st.download_button(
                    "⬇️ Download Excel File",
                    excel_data,
                    f"reranker_results_{metadata.get('timestamp', 'export')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            elif export_format == "CSV (Summary)":
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv_data,
                    f"summary_{metadata.get('timestamp', 'export')}.csv",
                    "text/csv"
                )

            elif export_format == "CSV (Detailed)":
                csv_data = df_detailed.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv_data,
                    f"detailed_{metadata.get('timestamp', 'export')}.csv",
                    "text/csv"
                )

            elif export_format == "JSON (Raw)":
                json_data = json.dumps({
                    "metadata": metadata,
                    "results": results
                }, indent=2)

                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    f"results_{metadata.get('timestamp', 'export')}.json",
                    "application/json"
                )

            st.success("✅ Export ready for download!")

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            st.exception(e)

    st.markdown("---")

    # Preview export data
    with st.expander("📋 Preview Export Data"):
        if export_format == "Excel (Multi-sheet)" or export_format == "CSV (Summary)":
            st.markdown("**Summary Table:**")
            st.dataframe(summary_df.head(10), use_container_width=True)

        if export_format == "Excel (Multi-sheet)" or export_format == "CSV (Detailed)":
            st.markdown("**Detailed Table (first 10 rows):**")
            st.dataframe(df_detailed.head(10), use_container_width=True)

        if export_format == "JSON (Raw)":
            st.markdown("**JSON Structure:**")
            st.json({
                "metadata": metadata,
                "results": [results[0]] if results else []
            })
