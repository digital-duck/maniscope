"""
Grid Search Page - Parameter Sensitivity Analysis for Maniscope

Interactive Streamlit page for exploring optimal k (neighbors) and α (alpha/hybrid weight)
parameters using 3D ECharts visualizations.

Purpose: Generate visualizations for arxiv paper section "A.2 Parameter Sensitivity Analysis"
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Tuple
from streamlit_echarts import st_echarts

# Add utils to path for release structure
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS, OUTPUT_DIRS, ensure_output_dirs
from utils.grid_search import run_single_experiment, load_dataset, parse_range
from utils.metrics import calculate_all_metrics

# Page configuration
st.set_page_config(page_title="Grid Search", layout="wide", page_icon="🔍")

# ============================================================================
# ECharts Helper Functions
# ============================================================================

def create_3d_surface_option(df: pd.DataFrame, metric: str, dataset_name: str) -> dict:
    """
    Create ECharts 3D surface plot option for (k, alpha) vs metric

    Args:
        df: Grid search results with columns ['k', 'alpha', metric]
        metric: Metric name (e.g., 'MRR', 'NDCG@3')
        dataset_name: Dataset name for title

    Returns:
        ECharts option dict for st_echarts()
    """
    # Pivot data for surface
    pivot = df.pivot(index='k', columns='alpha', values=metric)

    # Prepare data in format: [[x, y, z], ...]
    data = []
    k_vals = sorted(df['k'].unique())
    alpha_vals = sorted(df['alpha'].unique())

    for i, k in enumerate(k_vals):
        for j, alpha in enumerate(alpha_vals):
            z_val = pivot.loc[k, alpha] if not pd.isna(pivot.loc[k, alpha]) else 0
            data.append([i, j, float(z_val)])

    # Find optimal point
    max_idx = df[metric].idxmax()
    opt_k = int(df.loc[max_idx, 'k'])
    opt_alpha = float(df.loc[max_idx, 'alpha'])
    opt_val = float(df.loc[max_idx, metric])

    opt_k_idx = k_vals.index(opt_k)
    opt_alpha_idx = alpha_vals.index(opt_alpha)

    option = {
        'title': {
            'text': f'{metric} vs (k, α) - {dataset_name}',
            'left': 'center',
            'textStyle': {'fontSize': 18}
        },
        'tooltip': {
            'trigger': 'item',
            'formatter': '{c}'
        },
        'visualMap': {
            'min': float(df[metric].min()),
            'max': float(df[metric].max()),
            'inRange': {'color': ['#313695', '#4575b4', '#abd9e9', '#fee090', '#f46d43', '#a50026']},
            'calculable': True,
            'right': 10,
            'top': 'middle',
            'textStyle': {'color': '#000'}
        },
        'xAxis3D': {
            'type': 'category',
            'name': 'k (neighbors)',
            'data': [str(k) for k in k_vals]
        },
        'yAxis3D': {
            'type': 'category',
            'name': 'α (hybrid weight)',
            'data': [f"{v:.2f}" for v in alpha_vals]
        },
        'zAxis3D': {
            'type': 'value',
            'name': metric
        },
        'grid3D': {
            'viewControl': {
                'projection': 'perspective',
                'autoRotate': False,
                'distance': 200
            },
            'boxWidth': 100,
            'boxDepth': 100,
            'boxHeight': 60,
            'light': {
                'main': {'intensity': 1.2},
                'ambient': {'intensity': 0.3}
            }
        },
        'series': [
            {
                'type': 'scatter3D',
                'data': data,
                'symbolSize': 10,
                'itemStyle': {
                    'opacity': 0.8
                },
                'emphasis': {
                    'itemStyle': {
                        'color': '#ffff00'
                    }
                }
            },
            {
                'type': 'scatter3D',
                'data': [[opt_k_idx, opt_alpha_idx, float(opt_val)]],
                'symbolSize': 20,
                'itemStyle': {'color': 'red'},
                'label': {
                    'show': True,
                    'formatter': f'Optimal\\nk={opt_k}\\nα={opt_alpha:.2f}',
                    'textStyle': {
                        'fontSize': 14,
                        'fontWeight': 'bold',
                        'backgroundColor': 'rgba(255,255,255,0.8)',
                        'padding': [5, 8]
                    }
                }
            }
        ]
    }

    return option


def create_2d_heatmap_option(df: pd.DataFrame, metric: str, dataset_name: str) -> dict:
    """
    Create ECharts 2D heatmap for (k, alpha) vs metric

    Provides an alternative view to 3D surface for easier reading
    """
    pivot = df.pivot(index='k', columns='alpha', values=metric)

    k_vals = sorted(df['k'].unique())
    alpha_vals = sorted(df['alpha'].unique())

    # Prepare data: [[x_idx, y_idx, value], ...]
    data = []
    for i, k in enumerate(k_vals):
        for j, alpha in enumerate(alpha_vals):
            z_val = pivot.loc[k, alpha] if not pd.isna(pivot.loc[k, alpha]) else 0
            data.append([j, i, round(float(z_val), 4)])

    option = {
        'title': {
            'text': f'{metric} Heatmap - {dataset_name}',
            'left': 'center'
        },
        'tooltip': {
            'position': 'top',
            'formatter': '{c}'
        },
        'grid': {
            'height': '70%',
            'top': '15%',
            'left': '15%',
            'right': '15%'
        },
        'xAxis': {
            'type': 'category',
            'data': [f"{v:.2f}" for v in alpha_vals],
            'name': 'α (hybrid weight)',
            'nameLocation': 'middle',
            'nameGap': 30,
            'splitArea': {'show': True}
        },
        'yAxis': {
            'type': 'category',
            'data': [str(k) for k in k_vals],
            'name': 'k (neighbors)',
            'nameLocation': 'middle',
            'nameGap': 50,
            'splitArea': {'show': True}
        },
        'visualMap': {
            'min': float(df[metric].min()),
            'max': float(df[metric].max()),
            'calculable': True,
            'orient': 'horizontal',
            'left': 'center',
            'bottom': '5%',
            'inRange': {'color': ['#313695', '#4575b4', '#abd9e9', '#fee090', '#f46d43', '#a50026']}
        },
        'series': [{
            'type': 'heatmap',
            'data': data,
            'label': {
                'show': True,
                'formatter': '{@[2]}'
            },
            'emphasis': {
                'itemStyle': {
                    'shadowBlur': 10,
                    'shadowColor': 'rgba(0, 0, 0, 0.5)'
                }
            }
        }]
    }

    return option


# ============================================================================
# Analysis Functions
# ============================================================================

def display_summary_statistics(df: pd.DataFrame, metrics: list, dataset_name: str):
    """Display summary statistics and optimal parameters"""

    st.subheader("🏆 Optimal Parameters by Metric")

    # Create table of optimal parameters
    optimal_params = []
    for metric in metrics:
        if metric not in df.columns:
            continue

        max_idx = df[metric].idxmax()
        best_row = df.loc[max_idx]

        optimal_params.append({
            'Metric': metric,
            'Best k': int(best_row['k']),
            'Best α': f"{best_row['alpha']:.2f}",
            'Score': f"{best_row[metric]:.4f}",
            'Min': f"{df[metric].min():.4f}",
            'Max': f"{df[metric].max():.4f}",
            'StdDev': f"{df[metric].std():.4f}"
        })

    st.dataframe(pd.DataFrame(optimal_params), use_container_width=True)

    # Show sensitivity analysis
    st.subheader("📊 Parameter Sensitivity")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Impact of k (averaging across α)**")
        for metric in ['MRR', 'NDCG@3', 'P@3']:
            if metric in df.columns:
                k_impact = df.groupby('k')[metric].mean()
                best_k = k_impact.idxmax()
                improvement = (k_impact.max() - k_impact.min()) / k_impact.min() * 100
                st.metric(
                    f"{metric} - Best k",
                    value=f"{best_k}",
                    delta=f"{improvement:.1f}% range",
                    help=f"Best k={best_k} with avg {metric}={k_impact.max():.4f}"
                )

    with col2:
        st.markdown("**Impact of α (averaging across k)**")
        for metric in ['MRR', 'NDCG@3', 'P@3']:
            if metric in df.columns:
                alpha_impact = df.groupby('alpha')[metric].mean()
                best_alpha = alpha_impact.idxmax()
                improvement = (alpha_impact.max() - alpha_impact.min()) / alpha_impact.min() * 100
                st.metric(
                    f"{metric} - Best α",
                    value=f"{best_alpha:.2f}",
                    delta=f"{improvement:.1f}% range",
                    help=f"Best α={best_alpha:.2f} with avg {metric}={alpha_impact.max():.4f}"
                )


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.header("🔍 Grid Search - Parameter Sensitivity Analysis")

    st.info("👈 Configure grid search parameters in the sidebar and click **Run Grid Search** to start")

    # Initialize session state
    if 'grid_search_results' not in st.session_state:
        st.session_state.grid_search_results = None
    if 'grid_search_metadata' not in st.session_state:
        st.session_state.grid_search_metadata = None

    # ========================================================================
    # Sidebar Configuration
    # ========================================================================

    with st.sidebar:
        st.subheader("⚙️ Grid Search Configuration")

        # Dataset Selection
        # st.markdown("### 📊 Dataset")
        dataset_options = {d['name']: d for d in DATASETS}
        dataset_names = sorted(list(dataset_options.keys()))

        selected_dataset_name = st.selectbox(
            "Select Dataset",
            options=dataset_names,
            index=0,
            help="Choose a dataset for grid search"
        )

        selected_config = dataset_options[selected_dataset_name]
        st.caption(f"📝 {selected_config['description']}   (Queries: {selected_config['queries']})")


        # Quick Presets
        # st.markdown("### 🚀 Quick Presets")
        preset_options = ["Coarse Scan", "Fine Scan", "Alpha Focus", "Custom", ]
        preset = st.selectbox(
            "Choose a preset",
            options=preset_options,
            index=preset_options.index("Coarse Scan"),
            help="Select a preset or choose Custom to set your own ranges"
        )

        # Set defaults based on preset
        if preset == "Coarse Scan":
            default_k_min, default_k_max, default_k_step = 3, 15, 4
            default_alpha_min, default_alpha_max, default_alpha_step = 0.0, 1.0, 0.5
        elif preset == "Fine Scan":
            default_k_min, default_k_max, default_k_step = 3, 15, 1
            default_alpha_min, default_alpha_max, default_alpha_step = 0.0, 1.0, 0.1
        elif preset == "Alpha Focus":
            default_k_min, default_k_max, default_k_step = 5, 5, 1
            default_alpha_min, default_alpha_max, default_alpha_step = 0.0, 1.0, 0.05
        else:  # Custom
            default_k_min, default_k_max, default_k_step = 3, 15, 1
            default_alpha_min, default_alpha_max, default_alpha_step = 0.0, 1.0, 0.25

        # K Range Configuration
        st.markdown("### 🔢 k (Neighbors) Range")
        col1, col2, col3 = st.columns(3)
        with col1:
            k_min = st.number_input("Min", min_value=1, max_value=50, value=default_k_min, step=1, key="k_min")
        with col2:
            k_max = st.number_input("Max", min_value=1, max_value=50, value=default_k_max, step=1, key="k_max")
        with col3:
            k_step = st.number_input("Step", min_value=1, max_value=10, value=default_k_step, step=1, key="k_step")

        # Preview k values
        k_values = list(range(k_min, k_max + 1, k_step))
        st.caption(f"Will test {len(k_values)} values: {k_values}")

        # Alpha Range Configuration
        st.markdown("### 🎚️ α (Hybrid Weight) Range")
        col1, col2, col3 = st.columns(3)
        with col1:
            alpha_min = st.number_input("Min", min_value=0.0, max_value=1.0, value=default_alpha_min, step=0.05, format="%.2f", key="alpha_min")
        with col2:
            alpha_max = st.number_input("Max", min_value=0.0, max_value=1.0, value=default_alpha_max, step=0.05, format="%.2f", key="alpha_max")
        with col3:
            alpha_step = st.number_input("Step", min_value=0.01, max_value=0.5, value=default_alpha_step, step=0.01, format="%.2f", key="alpha_step")

        # Preview alpha values
        alpha_values = list(np.arange(alpha_min, alpha_max + alpha_step/2, alpha_step))
        alpha_values = [round(v, 10) for v in alpha_values]  # Round to avoid floating point errors
        st.caption(f"Will test {len(alpha_values)} values: {[f'{v:.2f}' for v in alpha_values]}")

        # Optional Query Limit
        # st.markdown("### 🎯 Query Limit")
        max_queries_options = ["All", "10", "25", "50", "100"]
        max_queries_str = st.selectbox(
            "Query Limit",
            options=max_queries_options,
            index=0,
            help="Limit number of queries for quick testing"
        )
        max_queries = None if max_queries_str == "All" else int(max_queries_str)

        # Total Experiments Display
        total_experiments = len(k_values) * len(alpha_values)
        st.info(f"**Total Experiments**: {total_experiments} (k × α combinations)")

        if max_queries:
            st.caption(f"Using first {max_queries} queries from dataset")

        # Run Button
        run_button = st.button(
            "▶️ Run Grid Search",
            type="primary",
            use_container_width=True,
            help="Start grid search with current parameters"
        )

        # Instructions at bottom of sidebar
        with st.expander("📖 Instructions", expanded=False):
            st.markdown("""
            ### Purpose
            Systematically explore **k** (neighbors) and **α** (hybrid weight)
            parameters to find optimal Maniscope configuration.

            ### Parameters
            **k (neighbors)**: Controls k-NN graph size (typical: 3-15)
            - Higher k = smoother paths
            - Lower k = more sensitive to local structure

            **α (hybrid weight)**: Balances geodesic vs. cosine (0.0 to 1.0)
            - α = 0.0: Pure geodesic (graph-based)
            - α = 1.0: Pure cosine (embedding-based)
            - α = 0.5: Balanced hybrid

            ### Quick Workflow
            1. Select dataset (start with Quick datasets)
            2. Choose preset or set custom ranges
            3. Click "▶️ Run Grid Search"
            4. Explore results in tabs (3D, heatmap, stats)
            5. Export for paper or further analysis

            ### Tips
            - **Coarse Scan** first to identify promising regions
            - **Fine Scan** to zoom in on optimal ranges
            - Use query limits for faster testing
            - Export results for reproducibility
            """)

    # ========================================================================
    # Main Content Area
    # ========================================================================

    # Handle run button
    if run_button:
        # Validate parameters
        if k_min > k_max:
            st.error("❌ k Min must be <= k Max")
            return
        if alpha_min > alpha_max:
            st.error("❌ α Min must be <= α Max")
            return
        if total_experiments > 1000:
            st.warning("⚠️ Large number of experiments may take a long time. Consider using a smaller range or preset.")

        # Load dataset
        try:
            dataset_short = selected_config.get('short', 'dataset')
            dataset_file = selected_config['file']

            # Try to load from standard location - FIXED for release structure
            data_dir = Path(__file__).parent.parent.parent / "data"
            dataset_path = data_dir / dataset_file

            if not dataset_path.exists():
                st.error(f"❌ Dataset file not found: {dataset_path}")
                st.info("💡 Go to Data Manager to ensure the dataset is loaded")
                return

            with open(dataset_path, 'r') as f:
                dataset_list = json.load(f)

            # Limit queries if specified
            if max_queries and max_queries < len(dataset_list):
                dataset_list = dataset_list[:max_queries]

            st.success(f"✅ Loaded {selected_dataset_name} ({len(dataset_list)} queries)")

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            return

        # Run grid search
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Progress tracking
        progress_bar = st.progress(0)
        status_container = st.empty()

        with st.status("Running grid search...", expanded=True) as status:
            for i, k in enumerate(k_values):
                for j, alpha in enumerate(alpha_values):
                    experiment_num = i * len(alpha_values) + j + 1

                    status_container.text(f"Experiment {experiment_num}/{total_experiments}: k={k}, α={alpha:.2f}")
                    progress_bar.progress(experiment_num / total_experiments)

                    try:
                        metrics = run_single_experiment(dataset_list, k, alpha, verbose=False)

                        result = {
                            'k': k,
                            'alpha': alpha,
                            **metrics
                        }
                        results.append(result)

                        # Show current metrics
                        st.write(f"k={k}, α={alpha:.2f}: MRR={metrics.get('MRR', 0):.4f}, NDCG@3={metrics.get('NDCG@3', 0):.4f}")

                    except Exception as e:
                        st.error(f"❌ Failed at k={k}, α={alpha:.2f}: {e}")
                        results.append({
                            'k': k,
                            'alpha': alpha,
                            'error': str(e)
                        })

            status.update(label="Grid search complete!", state="complete")

        progress_bar.progress(1.0)
        status_container.success(f"✅ Completed {len(results)} experiments")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Store in session state
        st.session_state.grid_search_results = df
        st.session_state.grid_search_metadata = {
            'dataset': selected_dataset_name,
            'timestamp': timestamp,
            'k_range': {'min': k_min, 'max': k_max, 'step': k_step},
            'alpha_range': {'min': alpha_min, 'max': alpha_max, 'step': alpha_step},
            'total_experiments': total_experiments,
            'num_queries': len(dataset_list)
        }

    # ========================================================================
    # Display Results (if available)
    # ========================================================================

    if st.session_state.grid_search_results is not None:
        df = st.session_state.grid_search_results
        metadata = st.session_state.grid_search_metadata

        st.markdown("---")
        st.header("📊 Results")

        # Show metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset", metadata['dataset'])
        with col2:
            st.metric("Total Experiments", metadata['total_experiments'])
        with col3:
            st.metric("Queries", metadata['num_queries'])

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs([
            "🔥 2D Heatmaps",
            "📈 Metric Comparison",
            "💾 Export"
        ])

        # Tab 1: 2D Heatmaps (moved from Tab 2)
        with tab1:
            st.subheader("2D Heatmap Visualizations")
            st.caption("Precise value reading with color-coded performance")

            metrics_to_plot = ['MRR', 'NDCG@3', 'NDCG@5', 'P@3', 'P@5', 'MAP']
            available_metrics = [m for m in metrics_to_plot if m in df.columns]

            # Create 2-column layout
            for idx in range(0, len(available_metrics), 2):
                cols = st.columns(2)

                for col_idx, metric in enumerate(available_metrics[idx:idx+2]):
                    with cols[col_idx]:
                        st.markdown(f"### {metric}")
                        try:
                            option = create_2d_heatmap_option(df, metric, metadata['dataset'])
                            st_echarts(options=option, height="400px", key=f"heatmap_{metric}")
                        except Exception as e:
                            st.error(f"Chart error: {str(e)}")
                            st.write("Debug info:", option if 'option' in locals() else "option not created")

        # Tab 2: Summary Statistics
        with tab2:
            metrics_to_plot = ['MRR', 'NDCG@3', 'NDCG@5', 'P@3', 'P@5', 'MAP']
            available_metrics = [m for m in metrics_to_plot if m in df.columns]
            display_summary_statistics(df, available_metrics, metadata['dataset'])

        # Tab 3: Export
        with tab3:
            st.subheader("💾 Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Export as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"grid_search_{metadata['dataset']}_{metadata['timestamp']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # Export as JSON with metadata
                export_data = {
                    'metadata': metadata,
                    'results': df.to_dict('records')
                }
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"grid_search_{metadata['dataset']}_{metadata['timestamp']}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col3:
                # Save to output directory - FIXED for release structure
                if st.button("💾 Save to Output Directory", use_container_width=True):
                    ensure_output_dirs()  # Ensure all output directories exist
                    output_dir = OUTPUT_DIRS["grid_search"]

                    csv_path = output_dir / f"grid_search_{metadata['dataset']}_{metadata['timestamp']}.csv"
                    json_path = output_dir / f"grid_search_{metadata['dataset']}_{metadata['timestamp']}.json"

                    df.to_csv(csv_path, index=False)
                    with open(json_path, 'w') as f:
                        json.dump(export_data, f, indent=2)

                    st.success(f"✅ Saved to {output_dir}")

            # Show raw data
            st.markdown("### 📋 Raw Data")
            st.dataframe(df, use_container_width=True)



if __name__ == "__main__":
    main()
