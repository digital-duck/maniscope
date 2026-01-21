"""
Visualization utilities for reranker evaluation results.

Provides plotting functions for:
- Metric comparisons (bar charts)
- Latency analysis (box plots, distributions)
- Pareto frontiers (accuracy vs latency)
- Per-query heatmaps
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def plot_metric_comparison(results: List[Dict], metric_name: str, model_names: List[str] = None) -> go.Figure:
    """
    Create bar chart comparing a specific metric across models.

    Args:
        results: List of result dicts with "models" or "rerankers" containing metrics
        metric_name: Metric to plot (e.g., "MRR", "NDCG@3")
        model_names: List of model names to include (None = all)

    Returns:
        Plotly figure
    """
    # Aggregate metrics per model
    model_metrics = {}

    for result in results:
        # Handle both "models" and "rerankers" keys for backward compatibility
        models_dict = result.get("models", result.get("rerankers", {}))
        for model_name, model_data in models_dict.items():
            if model_names and model_name not in model_names:
                continue

            if model_name not in model_metrics:
                model_metrics[model_name] = []

            metric_value = model_data.get("metrics", {}).get(metric_name, 0)
            model_metrics[model_name].append(metric_value)

    # Calculate averages
    avg_metrics = {
        model: np.mean(values) if values else 0
        for model, values in model_metrics.items()
    }

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(avg_metrics.keys()),
        y=list(avg_metrics.values()),
        text=[f"{v:.4f}" for v in avg_metrics.values()],
        textposition='auto',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title=f"{metric_name} Comparison",
        xaxis_title="Model",
        yaxis_title=metric_name,
        yaxis_range=[0, 1.1],
        showlegend=False,
        template="plotly_white",
        height=400
    )

    return fig


def plot_all_metrics_comparison(results: List[Dict], model_names: List[str] = None) -> go.Figure:
    """
    Create grouped bar chart showing all metrics for all models.

    Args:
        results: List of result dicts
        model_names: List of model names to include

    Returns:
        Plotly figure
    """
    metrics_to_plot = ["MRR", "NDCG@3", "NDCG@10", "P@1", "P@3", "MAP"]

    # Aggregate data
    data_for_plot = []

    for metric_name in metrics_to_plot:
        for result in results:
            # Handle both "models" and "rerankers" keys for backward compatibility
            models_dict = result.get("models", result.get("rerankers", {}))
            for model_name, model_data in models_dict.items():
                if model_names and model_name not in model_names:
                    continue

                metric_value = model_data.get("metrics", {}).get(metric_name, 0)
                data_for_plot.append({
                    "Model": model_name,
                    "Metric": metric_name,
                    "Value": metric_value
                })

    df = pd.DataFrame(data_for_plot)

    # Calculate averages
    df_avg = df.groupby(["Model", "Metric"])["Value"].mean().reset_index()

    # Create grouped bar chart
    fig = px.bar(
        df_avg,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="All Metrics Comparison",
        labels={"Value": "Score"},
        text="Value"
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 1.1],
        template="plotly_white",
        height=500
    )

    return fig


def plot_latency_distribution(results: List[Dict], model_names: List[str] = None) -> go.Figure:
    """
    Create box plot showing latency distribution for each model.

    Args:
        results: List of result dicts
        model_names: List of model names to include

    Returns:
        Plotly figure
    """
    data_for_plot = []

    for result in results:
        # Handle both "models" and "rerankers" keys for backward compatibility
        models_dict = result.get("models", result.get("rerankers", {}))
        for model_name, model_data in models_dict.items():
            if model_names and model_name not in model_names:
                continue

            latency = model_data.get("latency_ms", 0)
            data_for_plot.append({
                "Model": model_name,
                "Latency (ms)": latency
            })

    df = pd.DataFrame(data_for_plot)

    fig = px.box(
        df,
        x="Model",
        y="Latency (ms)",
        title="Latency Distribution by Model",
        points="all",
        color="Model"
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )

    return fig


def plot_pareto_frontier(results: List[Dict], metric_name: str = "NDCG@3",
                         model_names: List[str] = None) -> go.Figure:
    """
    Create scatter plot showing accuracy vs latency trade-off (Pareto frontier).

    Args:
        results: List of result dicts
        metric_name: Accuracy metric to use (default: NDCG@3)
        model_names: List of model names to include

    Returns:
        Plotly figure
    """
    # Aggregate metrics per model
    model_data = {}

    for result in results:
        # Handle both "models" and "rerankers" keys for backward compatibility
        models_dict = result.get("models", result.get("rerankers", {}))
        for model_name, data in models_dict.items():
            if model_names and model_name not in model_names:
                continue

            if model_name not in model_data:
                model_data[model_name] = {
                    "metrics": [],
                    "latencies": []
                }

            metric_value = data.get("metrics", {}).get(metric_name, 0)
            latency = data.get("latency_ms", 0)

            model_data[model_name]["metrics"].append(metric_value)
            model_data[model_name]["latencies"].append(latency)

    # Calculate averages
    plot_data = []
    for model_name, data in model_data.items():
        plot_data.append({
            "Model": model_name,
            "Accuracy": np.mean(data["metrics"]),
            "Latency (ms)": np.mean(data["latencies"])
        })

    df = pd.DataFrame(plot_data)

    # Create scatter plot
    fig = px.scatter(
        df,
        x="Latency (ms)",
        y="Accuracy",
        text="Model",
        title=f"Efficiency Frontier: {metric_name} vs Latency",
        labels={"Latency (ms)": "Latency (Lower is Better)",
                "Accuracy": f"{metric_name} (Higher is Better)"},
        size=[100] * len(df),  # Fixed size
        color="Model"
    )

    fig.update_traces(textposition='top center', textfont_size=12)
    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=True
    )

    # Add annotations for optimal regions
    fig.add_annotation(
        x=0.1, y=0.95,
        xref="paper", yref="paper",
        text="🎯 Optimal: Top-Left<br>(High Accuracy, Low Latency)",
        showarrow=False,
        bgcolor="lightgreen",
        opacity=0.8
    )

    return fig


def plot_per_query_heatmap(results: List[Dict], metric_name: str = "NDCG@3",
                           model_names: List[str] = None, max_queries: int = 50) -> go.Figure:
    """
    Create heatmap showing per-query performance across models.

    Args:
        results: List of result dicts
        metric_name: Metric to visualize
        model_names: List of model names to include
        max_queries: Maximum queries to show (for readability)

    Returns:
        Plotly figure
    """
    # Prepare data matrix
    queries_to_show = results[:max_queries]

    if not model_names:
        # Get all model names from first result
        # Handle both "models" and "rerankers" keys for backward compatibility
        models_dict = queries_to_show[0].get("models", queries_to_show[0].get("rerankers", {}))
        model_names = list(models_dict.keys())

    # Build matrix: rows = queries, cols = models
    matrix = []
    query_ids = []

    for result in queries_to_show:
        row = []
        query_ids.append(result.get("query_id", "?")[:20])

        # Handle both "models" and "rerankers" keys for backward compatibility
        models_dict = result.get("models", result.get("rerankers", {}))
        for model_name in model_names:
            model_data = models_dict.get(model_name, {})
            metric_value = model_data.get("metrics", {}).get(metric_name, 0)
            row.append(metric_value)

        matrix.append(row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=model_names,
        y=query_ids,
        colorscale='RdYlGn',
        text=[[f"{val:.3f}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title=metric_name)
    ))

    fig.update_layout(
        title=f"Per-Query {metric_name} Scores",
        xaxis_title="Model",
        yaxis_title="Query ID",
        template="plotly_white",
        height=max(600, len(queries_to_show) * 15)
    )

    return fig


def create_summary_table(results: List[Dict], model_names: List[str] = None) -> pd.DataFrame:
    """
    Create summary table with all metrics for all models.

    Args:
        results: List of result dicts
        model_names: List of model names to include

    Returns:
        DataFrame with summary statistics
    """
    metrics_list = ["MRR", "NDCG@1", "NDCG@3", "NDCG@10", "P@1", "P@3", "P@10", "R@10", "MAP"]

    summary_data = []

    # Collect all metrics per model
    model_metrics = {}

    for result in results:
        # Handle both "models" and "rerankers" keys for backward compatibility
        models_dict = result.get("models", result.get("rerankers", {}))
        for model_name, model_data in models_dict.items():
            if model_names and model_name not in model_names:
                continue

            if model_name not in model_metrics:
                model_metrics[model_name] = {metric: [] for metric in metrics_list}
                model_metrics[model_name]["latency"] = []

            for metric in metrics_list:
                value = model_data.get("metrics", {}).get(metric, 0)
                model_metrics[model_name][metric].append(value)

            latency = model_data.get("latency_ms", 0)
            model_metrics[model_name]["latency"].append(latency)

    # Calculate averages
    for model_name, metrics in model_metrics.items():
        row = {"Model": model_name}

        for metric in metrics_list:
            row[metric] = np.mean(metrics[metric]) if metrics[metric] else 0

        row["Avg Latency (ms)"] = np.mean(metrics["latency"]) if metrics["latency"] else 0
        row["Queries/sec"] = 1000 / row["Avg Latency (ms)"] if row["Avg Latency (ms)"] > 0 else 0

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Round for display
    numeric_cols = [col for col in df.columns if col != "Model"]
    for col in numeric_cols:
        if "Latency" in col or "Queries" in col:
            df[col] = df[col].round(2)
        else:
            df[col] = df[col].round(4)

    return df


def plot_metric_over_queries(results: List[Dict], metric_name: str = "NDCG@3",
                             model_names: List[str] = None) -> go.Figure:
    """
    Plot metric values over query sequence (to see if performance varies).

    Args:
        results: List of result dicts
        metric_name: Metric to plot
        model_names: List of model names to include

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if not model_names:
        # Handle both "models" and "rerankers" keys for backward compatibility
        models_dict = results[0].get("models", results[0].get("rerankers", {}))
        model_names = list(models_dict.keys())

    for model_name in model_names:
        metric_values = []

        for result in results:
            # Handle both "models" and "rerankers" keys for backward compatibility
            models_dict = result.get("models", result.get("rerankers", {}))
            model_data = models_dict.get(model_name, {})
            value = model_data.get("metrics", {}).get(metric_name, 0)
            metric_values.append(value)

        fig.add_trace(go.Scatter(
            y=metric_values,
            mode='lines+markers',
            name=model_name,
            line=dict(width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title=f"{metric_name} Across Queries",
        xaxis_title="Query Index",
        yaxis_title=metric_name,
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    return fig


# Test if run directly
if __name__ == "__main__":
    # Create sample data
    sample_results = [
        {
            "query_id": "1",
            "models": {
                "BGE-M3": {
                    "metrics": {"MRR": 1.0, "NDCG@3": 0.95, "P@1": 1.0},
                    "latency_ms": 25
                },
                "Qwen-1.5B": {
                    "metrics": {"MRR": 0.85, "NDCG@3": 0.90, "P@1": 1.0},
                    "latency_ms": 120
                }
            }
        },
        {
            "query_id": "2",
            "models": {
                "BGE-M3": {
                    "metrics": {"MRR": 0.5, "NDCG@3": 0.70, "P@1": 0.0},
                    "latency_ms": 23
                },
                "Qwen-1.5B": {
                    "metrics": {"MRR": 1.0, "NDCG@3": 0.95, "P@1": 1.0},
                    "latency_ms": 115
                }
            }
        }
    ]

    print("Testing visualization utilities...")

    # Test summary table
    df = create_summary_table(sample_results)
    print("\nSummary Table:")
    print(df.to_string())

    # Test metric comparison
    fig = plot_metric_comparison(sample_results, "NDCG@3")
    print(f"\nCreated metric comparison chart: {type(fig)}")

    # Test pareto frontier
    fig = plot_pareto_frontier(sample_results)
    print(f"Created pareto frontier chart: {type(fig)}")

    print("\n✅ All visualization functions working!")
