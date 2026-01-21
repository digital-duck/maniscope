#!/usr/bin/env python3
"""
Grid Search for Maniscope Hyperparameters

Systematically searches k (neighbors) and α (hybrid weight) parameters
to find optimal configuration for each dataset.

Usage:
    python grid_search.py --dataset aorb --k-range 3:20:2 --alpha-range 0.0:1.0:0.1
    python grid_search.py --dataset scifact --k-range 5:15:5 --alpha-range 0.0:0.5:0.1
"""

import click
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from maniscope import ManiscopeEngine
from metrics import calculate_all_metrics


def parse_range(range_str: str, is_float: bool = False) -> List:
    """
    Parse range string like "3:20:2" into list of values.

    Args:
        range_str: Format "min:max:step"
        is_float: If True, parse as floats, else as ints

    Returns:
        List of values in range

    Examples:
        >>> parse_range("3:10:2", is_float=False)
        [3, 5, 7, 9]
        >>> parse_range("0.0:1.0:0.5", is_float=True)
        [0.0, 0.5, 1.0]
    """
    parts = range_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Range must be 'min:max:step', got: {range_str}")

    start, stop, step = parts

    if is_float:
        start, stop, step = float(start), float(stop), float(step)
        # Use arange with small epsilon to include endpoint
        values = np.arange(start, stop + step/2, step)
        return [round(v, 10) for v in values]  # Round to avoid floating point errors
    else:
        start, stop, step = int(start), int(stop), int(step)
        return list(range(start, stop + 1, step))


def load_dataset(dataset_name: str) -> Tuple[List[dict], Path]:
    """
    Load dataset by name.

    Args:
        dataset_name: Dataset name (e.g., "aorb", "scifact", "msmarco")

    Returns:
        Tuple of (dataset_list, dataset_path)
    """
    # Map short names to file names
    dataset_map = {
        'aorb': 'dataset-aorb.json',
        'aorb-10': 'dataset-aorb-10.json',
        'scifact': 'dataset-scifact.json',
        'scifact-10': 'dataset-scifact-10.json',
        'msmarco': 'dataset-msmarco.json',
        'msmarco-10': 'dataset-msmarco-10.json',
        'trec-covid': 'dataset-trec-covid.json',
        'trec-covid-10': 'dataset-trec-covid-10.json',
        'arguana': 'dataset-arguana.json',
        'arguana-10': 'dataset-arguana-10.json',
        'fiqa': 'dataset-fiqa.json',
        'fiqa-10': 'dataset-fiqa-10.json',
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")

    # Find data directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    dataset_file = data_dir / dataset_map[dataset_name]

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    return dataset, dataset_file


def run_single_experiment(dataset: List[dict], k: int, alpha: float,
                         verbose: bool = False) -> Dict[str, float]:
    """
    Run Maniscope with specific k and alpha parameters.

    Args:
        dataset: List of query-document pairs
        k: Number of neighbors for k-NN graph
        alpha: Hybrid weight (0=pure geodesic, 1=pure cosine)
        verbose: Print progress

    Returns:
        Dict with aggregated metrics (MRR, NDCG@3, etc.)
    """
    # Initialize model
    model = ManiscopeEngine(
        model_name='all-MiniLM-L6-v2',
        k=k,
        alpha=alpha,
        verbose=False,
        device='cpu',  # Use CPU for consistency
        local_files_only=True
    )

    all_metrics = []

    for i, item in enumerate(dataset):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processing query {i+1}/{len(dataset)}...", end='\r')

        query = item['query']
        docs = item['docs']
        relevance_map = item['relevance_map']

        # Fit model and search
        model.fit(docs)
        results = model.search_maniscope_detailed(query, top_n=len(docs), coarse_multiplier=3)

        # Extract rankings
        rankings = [r['doc_id'] for r in results]

        # Calculate metrics
        metrics = calculate_all_metrics(rankings, relevance_map)
        all_metrics.append(metrics)

    if verbose:
        print()  # Clear progress line

    # Aggregate metrics (average across queries)
    aggregated = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        aggregated[metric_name] = float(np.mean(values))

    return aggregated


@click.command()
@click.option('--dataset', required=True,
              help='Dataset name (e.g., aorb, scifact, msmarco)')
@click.option('--k-range', required=True,
              help='Range for k parameter (format: min:max:step, e.g., 3:20:2)')
@click.option('--alpha-range', required=True,
              help='Range for alpha parameter (format: min:max:step, e.g., 0.0:1.0:0.1)')
@click.option('--max-queries', default=None, type=int,
              help='Limit number of queries (for quick testing)')
@click.option('--output-dir', default='output/grid_search',
              help='Directory to save results')
@click.option('--verbose/--quiet', default=True,
              help='Print progress')
def grid_search(dataset: str, k_range: str, alpha_range: str,
                max_queries: int, output_dir: str, verbose: bool):
    """
    Grid search for optimal Maniscope hyperparameters.

    Systematically tests combinations of k (neighbors) and α (hybrid weight)
    to find configuration that maximizes retrieval accuracy.

    Examples:

        # Quick coarse scan on AorB (3 k values × 5 alpha values = 15 experiments)
        python grid_search.py --dataset aorb-10 --k-range 3:15:6 --alpha-range 0.0:1.0:0.25

        # Fine-grained scan on SciFact
        python grid_search.py --dataset scifact-10 --k-range 3:20:1 --alpha-range 0.0:1.0:0.05

        # Test only 5 queries for very quick validation
        python grid_search.py --dataset aorb --k-range 5:10:5 --alpha-range 0.3:0.7:0.2 --max-queries 5
    """
    click.echo(f"\n{'='*70}")
    click.echo(f"Maniscope Hyperparameter Grid Search")
    click.echo(f"{'='*70}\n")

    # Parse ranges
    try:
        k_values = parse_range(k_range, is_float=False)
        alpha_values = parse_range(alpha_range, is_float=True)
    except ValueError as e:
        click.echo(f"❌ Error parsing range: {e}", err=True)
        sys.exit(1)

    click.echo(f"Dataset: {dataset}")
    click.echo(f"k values: {k_values} ({len(k_values)} values)")
    click.echo(f"α values: {alpha_values} ({len(alpha_values)} values)")
    click.echo(f"Total experiments: {len(k_values) * len(alpha_values)}")
    click.echo()

    # Load dataset
    try:
        dataset_list, dataset_path = load_dataset(dataset)
        click.echo(f"✓ Loaded dataset: {dataset_path}")
        click.echo(f"  Queries: {len(dataset_list)}")

        if max_queries and max_queries < len(dataset_list):
            dataset_list = dataset_list[:max_queries]
            click.echo(f"  Limited to: {max_queries} queries")

        click.echo()
    except Exception as e:
        click.echo(f"❌ Error loading dataset: {e}", err=True)
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run grid search
    results = []
    total_experiments = len(k_values) * len(alpha_values)
    experiment_num = 0

    click.echo("Running grid search...")
    click.echo("-" * 70)

    for k in k_values:
        for alpha in alpha_values:
            experiment_num += 1

            if verbose:
                click.echo(f"\n[{experiment_num}/{total_experiments}] k={k}, α={alpha:.3f}")

            try:
                metrics = run_single_experiment(dataset_list, k, alpha, verbose=verbose)

                result = {
                    'k': k,
                    'alpha': alpha,
                    **metrics
                }
                results.append(result)

                if verbose:
                    click.echo(f"  MRR: {metrics['MRR']:.4f}, "
                             f"NDCG@3: {metrics.get('NDCG@3', 0):.4f}, "
                             f"MAP: {metrics.get('MAP', 0):.4f}")

            except Exception as e:
                click.echo(f"  ❌ Failed: {e}", err=True)
                results.append({
                    'k': k,
                    'alpha': alpha,
                    'error': str(e)
                })

    click.echo("\n" + "=" * 70)
    click.echo("Grid Search Complete!")
    click.echo("=" * 70 + "\n")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Find best parameters by MRR
    if 'MRR' in df.columns:
        best_idx = df['MRR'].idxmax()
        best_result = df.iloc[best_idx]

        click.echo("🏆 Best Parameters:")
        click.echo(f"  k = {best_result['k']}")
        click.echo(f"  α = {best_result['alpha']:.3f}")
        click.echo(f"\n  Metrics:")
        click.echo(f"    MRR:     {best_result['MRR']:.4f}")
        if 'NDCG@3' in df.columns:
            click.echo(f"    NDCG@3:  {best_result['NDCG@3']:.4f}")
        if 'NDCG@10' in df.columns:
            click.echo(f"    NDCG@10: {best_result['NDCG@10']:.4f}")
        if 'MAP' in df.columns:
            click.echo(f"    MAP:     {best_result['MAP']:.4f}")
        click.echo()

        # Show top 5 configurations
        click.echo("📊 Top 5 Configurations:")
        top5 = df.nlargest(5, 'MRR')[['k', 'alpha', 'MRR', 'NDCG@3', 'MAP']]
        click.echo(top5.to_string(index=False))
        click.echo()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"grid_search_{dataset}_{timestamp}.json"
    csv_file = output_path / f"grid_search_{dataset}_{timestamp}.csv"

    # Save JSON
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': {
                'dataset': dataset,
                'dataset_path': str(dataset_path),
                'num_queries': len(dataset_list),
                'k_range': k_range,
                'alpha_range': alpha_range,
                'k_values': k_values,
                'alpha_values': alpha_values,
                'total_experiments': total_experiments,
                'timestamp': timestamp
            },
            'results': results
        }, f, indent=2)

    # Save CSV
    df.to_csv(csv_file, index=False)

    click.echo(f"💾 Results saved:")
    click.echo(f"  JSON: {results_file}")
    click.echo(f"  CSV:  {csv_file}")
    click.echo()

    # Summary statistics
    if 'MRR' in df.columns:
        click.echo("📈 Summary Statistics:")
        click.echo(f"  MRR - Min: {df['MRR'].min():.4f}, "
                 f"Max: {df['MRR'].max():.4f}, "
                 f"Mean: {df['MRR'].mean():.4f}, "
                 f"Std: {df['MRR'].std():.4f}")

        # Best k value (average across alpha)
        best_k = df.groupby('k')['MRR'].mean().idxmax()
        click.echo(f"\n  Best k (averaged across α): {best_k}")

        # Best alpha value (average across k)
        best_alpha = df.groupby('alpha')['MRR'].mean().idxmax()
        click.echo(f"  Best α (averaged across k): {best_alpha:.3f}")

    click.echo("\n✅ Done!")


if __name__ == '__main__':
    grid_search()
