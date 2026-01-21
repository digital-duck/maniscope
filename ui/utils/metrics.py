"""
Information Retrieval Metrics for Reranker Evaluation

Implements standard IR metrics:
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Precision@K
- Recall@K
- MAP (Mean Average Precision)
"""

import numpy as np
from typing import Dict, List, Union


def calculate_mrr(rankings: List[int], relevance_map: Dict[str, int]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR = 1 / rank_of_first_relevant_doc

    Args:
        rankings: List of document indices in ranked order [best, 2nd, 3rd, ...]
        relevance_map: Dict mapping doc index (as string) to relevance score (1=relevant, 0=not)

    Returns:
        MRR score (0.0 to 1.0). Returns 0.0 if no relevant docs found.

    Example:
        >>> rankings = [2, 0, 1]  # Doc 2 ranked first, doc 0 second, doc 1 third
        >>> relevance_map = {"0": 1, "1": 0, "2": 0}  # Only doc 0 is relevant
        >>> calculate_mrr(rankings, relevance_map)
        0.5  # Relevant doc is at rank 2, so MRR = 1/2
    """
    for rank, doc_idx in enumerate(rankings, start=1):
        if relevance_map.get(str(doc_idx), 0) > 0:
            return 1.0 / rank
    return 0.0


def calculate_dcg(rankings: List[int], relevance_map: Dict[str, int], k: int = None) -> float:
    """
    Calculate Discounted Cumulative Gain.

    DCG@k = sum(rel_i / log2(i+1)) for i in 1..k

    Args:
        rankings: List of document indices in ranked order
        relevance_map: Dict mapping doc index (as string) to relevance score
        k: Cutoff rank (if None, uses all documents)

    Returns:
        DCG score (non-negative float)
    """
    if k is None:
        k = len(rankings)

    dcg = 0.0
    for i, doc_idx in enumerate(rankings[:k], start=1):
        rel = relevance_map.get(str(doc_idx), 0)
        dcg += rel / np.log2(i + 1)

    return dcg


def calculate_ndcg(rankings: List[int], relevance_map: Dict[str, int], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.

    NDCG@k = DCG@k / IDCG@k

    Normalizes DCG by the ideal DCG (documents sorted by relevance).

    Args:
        rankings: List of document indices in ranked order
        relevance_map: Dict mapping doc index (as string) to relevance score
        k: Cutoff rank (if None, uses all documents)

    Returns:
        NDCG score (0.0 to 1.0). Returns 0.0 if no relevant documents.

    Example:
        >>> rankings = [0, 1, 2]  # Perfect ranking
        >>> relevance_map = {"0": 3, "1": 2, "2": 1}  # Graded relevance
        >>> calculate_ndcg(rankings, relevance_map)
        1.0  # Perfect ranking achieves NDCG = 1.0
    """
    if k is None:
        k = len(rankings)

    # Actual DCG
    dcg = calculate_dcg(rankings, relevance_map, k)

    # Ideal DCG (sort by relevance descending)
    ideal_rankings = sorted(
        range(len(relevance_map)),
        key=lambda i: relevance_map.get(str(i), 0),
        reverse=True
    )
    idcg = calculate_dcg(ideal_rankings, relevance_map, k)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(rankings: List[int], relevance_map: Dict[str, int], k: int) -> float:
    """
    Calculate Precision@K.

    P@K = (# relevant docs in top K) / K

    Args:
        rankings: List of document indices in ranked order
        relevance_map: Dict mapping doc index (as string) to relevance score
        k: Cutoff rank

    Returns:
        Precision@K score (0.0 to 1.0)

    Example:
        >>> rankings = [0, 2, 1]  # Docs ranked: 0, 2, 1
        >>> relevance_map = {"0": 1, "1": 1, "2": 0}  # Docs 0 and 1 are relevant
        >>> calculate_precision_at_k(rankings, relevance_map, k=2)
        0.5  # 1 relevant doc in top 2, so P@2 = 1/2
    """
    if k <= 0:
        return 0.0

    relevant_in_top_k = sum(
        1 for doc_idx in rankings[:k]
        if relevance_map.get(str(doc_idx), 0) > 0
    )
    return relevant_in_top_k / k


def calculate_recall_at_k(rankings: List[int], relevance_map: Dict[str, int], k: int) -> float:
    """
    Calculate Recall@K.

    R@K = (# relevant docs in top K) / (total # relevant docs)

    Args:
        rankings: List of document indices in ranked order
        relevance_map: Dict mapping doc index (as string) to relevance score
        k: Cutoff rank

    Returns:
        Recall@K score (0.0 to 1.0). Returns 0.0 if no relevant documents exist.

    Example:
        >>> rankings = [0, 2, 1]  # Docs ranked: 0, 2, 1
        >>> relevance_map = {"0": 1, "1": 1, "2": 0}  # 2 relevant docs total
        >>> calculate_recall_at_k(rankings, relevance_map, k=1)
        0.5  # Found 1 out of 2 relevant docs in top 1, so R@1 = 1/2
    """
    total_relevant = sum(1 for score in relevance_map.values() if score > 0)
    if total_relevant == 0:
        return 0.0

    relevant_in_top_k = sum(
        1 for doc_idx in rankings[:k]
        if relevance_map.get(str(doc_idx), 0) > 0
    )
    return relevant_in_top_k / total_relevant


def calculate_map(rankings: List[int], relevance_map: Dict[str, int]) -> float:
    """
    Calculate Mean Average Precision.

    MAP = mean(precision@k for each relevant doc position k)

    Args:
        rankings: List of document indices in ranked order
        relevance_map: Dict mapping doc index (as string) to relevance score

    Returns:
        MAP score (0.0 to 1.0). Returns 0.0 if no relevant documents.

    Example:
        >>> rankings = [0, 2, 1]  # Docs ranked: 0, 2, 1
        >>> relevance_map = {"0": 1, "1": 1, "2": 0}  # Docs 0 and 1 relevant
        >>> calculate_map(rankings, relevance_map)
        0.833...  # AP = (1/1 + 2/3) / 2 = 0.833
    """
    precisions = []
    num_relevant = 0

    for k, doc_idx in enumerate(rankings, start=1):
        if relevance_map.get(str(doc_idx), 0) > 0:
            num_relevant += 1
            precision_at_k = num_relevant / k
            precisions.append(precision_at_k)

    return np.mean(precisions) if precisions else 0.0


def calculate_all_metrics(rankings: List[int], relevance_map: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate all standard IR metrics at once for efficiency.

    Computes:
    - MRR (Mean Reciprocal Rank)
    - NDCG@1, NDCG@3, NDCG@10 (Normalized Discounted Cumulative Gain)
    - P@1, P@3, P@10 (Precision at K)
    - R@10 (Recall at 10)
    - MAP (Mean Average Precision)

    Args:
        rankings: List of document indices in ranked order [best, 2nd, 3rd, ...]
        relevance_map: Dict mapping doc index (as string) to relevance score

    Returns:
        Dict with metric names as keys and scores as values

    Example:
        >>> rankings = [0, 1, 2, 3, 4]
        >>> relevance_map = {"0": 1, "1": 0, "2": 1, "3": 0, "4": 0}
        >>> metrics = calculate_all_metrics(rankings, relevance_map)
        >>> print(f"MRR: {metrics['MRR']:.3f}, NDCG@3: {metrics['NDCG@3']:.3f}")
        MRR: 1.000, NDCG@3: 0.613
    """
    return {
        "MRR": calculate_mrr(rankings, relevance_map),
        "NDCG@1": calculate_ndcg(rankings, relevance_map, k=1),
        "NDCG@3": calculate_ndcg(rankings, relevance_map, k=3),
        "NDCG@10": calculate_ndcg(rankings, relevance_map, k=10),
        "P@1": calculate_precision_at_k(rankings, relevance_map, k=1),
        "P@3": calculate_precision_at_k(rankings, relevance_map, k=3),
        "P@10": calculate_precision_at_k(rankings, relevance_map, k=10),
        "R@10": calculate_recall_at_k(rankings, relevance_map, k=10),
        "MAP": calculate_map(rankings, relevance_map)
    }


def calculate_metrics_aggregate(all_results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple queries for each model.

    Args:
        all_results: List of result dicts, each containing:
            {
                "query_id": "...",
                "models": {
                    "BGE-M3": {"rankings": [...], "metrics": {...}},
                    "Qwen": {"rankings": [...], "metrics": {...}},
                    ...
                }
            }

    Returns:
        Dict mapping model names to aggregated metrics:
        {
            "BGE-M3": {"MRR": 0.75, "NDCG@3": 0.68, ...},
            "Qwen": {"MRR": 0.82, "NDCG@3": 0.71, ...}
        }
    """
    model_metrics = {}

    # Collect all metrics for each model
    for result in all_results:
        for model_name, model_data in result.get("models", {}).items():
            if "metrics" not in model_data:
                continue

            if model_name not in model_metrics:
                model_metrics[model_name] = {metric: [] for metric in model_data["metrics"]}

            for metric, value in model_data["metrics"].items():
                model_metrics[model_name][metric].append(value)

    # Calculate mean for each metric
    aggregated = {}
    for model_name, metrics in model_metrics.items():
        aggregated[model_name] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in metrics.items()
        }

    return aggregated


# Quick test
if __name__ == "__main__":
    # Test with simple example
    rankings = [0, 2, 1]  # Doc 0 first, doc 2 second, doc 1 third
    relevance_map = {"0": 1, "1": 1, "2": 0}  # Docs 0 and 1 are relevant

    print("Test Rankings:", rankings)
    print("Relevance Map:", relevance_map)
    print("\nMetrics:")

    metrics = calculate_all_metrics(rankings, relevance_map)
    for name, value in metrics.items():
        print(f"  {name:10s}: {value:.4f}")
