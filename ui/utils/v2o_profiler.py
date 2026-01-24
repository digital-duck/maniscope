"""
V2O Optimization Profiler and Cache Manager

Provides tools to:
1. Monitor v2o cache hit rates and performance gains
2. Compare baseline vs v2o versions with detailed timing
3. Manage cache (clear, warmup, inspect)
4. Profile reranker performance for benchmarking

Usage:
    from utils.v2o_profiler import V2OProfiler, compare_baseline_vs_v2o

    # Profile a single reranker
    profiler = V2OProfiler(model_name="HNSW_v2o")
    profiler.profile_query(model, query, docs)
    profiler.print_stats()

    # Compare baseline vs v2o
    results = compare_baseline_vs_v2o("HNSW", query, docs, num_runs=10)
    results.print_comparison()
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib


@dataclass
class QueryProfile:
    """Profile data for a single query execution."""
    query: str
    num_docs: int
    latency_ms: float
    cache_hit: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProfileStats:
    """Aggregated profiling statistics."""
    model_name: str
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    cold_start_latency_ms: Optional[float] = None
    warm_latency_ms: Optional[float] = None
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.cache_hits / self.total_queries) * 100

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def speedup_ratio(self) -> Optional[float]:
        """Speedup ratio from cold to warm start."""
        if self.cold_start_latency_ms and self.warm_latency_ms:
            if self.warm_latency_ms > 0:
                return self.cold_start_latency_ms / self.warm_latency_ms
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "cold_start_latency_ms": round(self.cold_start_latency_ms, 2) if self.cold_start_latency_ms else None,
            "warm_latency_ms": round(self.warm_latency_ms, 2) if self.warm_latency_ms else None,
            "speedup_ratio": round(self.speedup_ratio, 2) if self.speedup_ratio else None
        }


class V2OProfiler:
    """
    Profiler for v2o optimized rerankers.

    Tracks cache hits, latency, and performance improvements.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.queries: List[QueryProfile] = []
        self.stats = ProfileStats(model_name=model_name)

    def profile_query(self, model: Any, query: str, docs: List[str],
                     run_func: Optional[callable] = None) -> Tuple[np.ndarray, float]:
        """
        Profile a single query execution.

        Args:
            model: Model instance or dict
            query: Query text
            docs: List of documents
            run_func: Optional custom run function (defaults to run_reranker)

        Returns:
            Tuple of (scores, latency_ms)
        """
        from utils.models import run_reranker

        if run_func is None:
            run_func = run_reranker

        # Check if this would be a cache hit
        cache_hit = self._check_cache_hit(model, query, docs)

        # Time the execution
        start_time = time.perf_counter()
        scores = run_func(model, query, docs)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Record profile
        profile = QueryProfile(
            query=query[:50] + "..." if len(query) > 50 else query,
            num_docs=len(docs),
            latency_ms=latency_ms,
            cache_hit=cache_hit
        )
        self.queries.append(profile)

        # Update stats
        self.stats.total_queries += 1
        self.stats.total_latency_ms += latency_ms
        self.stats.min_latency_ms = min(self.stats.min_latency_ms, latency_ms)
        self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)

        if cache_hit:
            self.stats.cache_hits += 1
            if self.stats.warm_latency_ms is None:
                self.stats.warm_latency_ms = latency_ms
            else:
                # Update to minimum warm latency
                self.stats.warm_latency_ms = min(self.stats.warm_latency_ms, latency_ms)
        else:
            self.stats.cache_misses += 1
            if self.stats.cold_start_latency_ms is None:
                self.stats.cold_start_latency_ms = latency_ms

        return scores, latency_ms

    def _check_cache_hit(self, model: Any, query: str, docs: List[str]) -> bool:
        """Check if query would hit cache (for v2o models)."""
        if not isinstance(model, dict):
            return False

        query_cache = model.get("query_cache", {})
        if not query_cache:
            return False

        # Create cache key matching the model's implementation
        cache_key = hashlib.md5((query + "||".join(docs)).encode()).hexdigest()
        return cache_key in query_cache

    def print_stats(self):
        """Print profiling statistics to console."""
        print(f"\n{'='*60}")
        print(f"V2O Profiling Results: {self.model_name}")
        print(f"{'='*60}")
        print(f"Total Queries:        {self.stats.total_queries}")
        print(f"Cache Hits:           {self.stats.cache_hits} ({self.stats.cache_hit_rate:.1f}%)")
        print(f"Cache Misses:         {self.stats.cache_misses}")
        print(f"\nLatency Statistics:")
        print(f"  Average:            {self.stats.avg_latency_ms:.2f} ms")
        print(f"  Min:                {self.stats.min_latency_ms:.2f} ms")
        print(f"  Max:                {self.stats.max_latency_ms:.2f} ms")

        if self.stats.cold_start_latency_ms:
            print(f"  Cold Start:         {self.stats.cold_start_latency_ms:.2f} ms")
        if self.stats.warm_latency_ms:
            print(f"  Warm (cached):      {self.stats.warm_latency_ms:.2f} ms")
        if self.stats.speedup_ratio:
            print(f"  Speedup:            {self.stats.speedup_ratio:.1f}×")
        print(f"{'='*60}\n")

    def save_results(self, output_path: Path):
        """Save profiling results to JSON file."""
        results = {
            "stats": self.stats.to_dict(),
            "queries": [
                {
                    "query": q.query,
                    "num_docs": q.num_docs,
                    "latency_ms": round(q.latency_ms, 2),
                    "cache_hit": q.cache_hit,
                    "timestamp": q.timestamp
                }
                for q in self.queries
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Saved profiling results to {output_path}")


@dataclass
class ComparisonResult:
    """Results from comparing baseline vs v2o versions."""
    model_name: str
    baseline_stats: ProfileStats
    v2o_stats: ProfileStats

    @property
    def avg_speedup(self) -> float:
        """Average speedup from baseline to v2o."""
        if self.v2o_stats.avg_latency_ms > 0:
            return self.baseline_stats.avg_latency_ms / self.v2o_stats.avg_latency_ms
        return 0.0

    @property
    def cache_benefit(self) -> float:
        """Speedup specifically from caching (cold v2o vs warm v2o)."""
        if self.v2o_stats.speedup_ratio:
            return self.v2o_stats.speedup_ratio
        return 1.0

    def print_comparison(self):
        """Print detailed comparison."""
        print(f"\n{'='*70}")
        print(f"Baseline vs V2O Comparison: {self.model_name}")
        print(f"{'='*70}")
        print(f"{'Metric':<30} {'Baseline':<15} {'V2O':<15} {'Improvement':<10}")
        print(f"{'-'*70}")

        baseline_avg = self.baseline_stats.avg_latency_ms
        v2o_avg = self.v2o_stats.avg_latency_ms
        avg_improvement = f"{self.avg_speedup:.1f}×" if self.avg_speedup > 1 else "-"

        print(f"{'Avg Latency (ms)':<30} {baseline_avg:<15.2f} {v2o_avg:<15.2f} {avg_improvement:<10}")

        # Cold start comparison
        if self.v2o_stats.cold_start_latency_ms:
            cold_improvement = baseline_avg / self.v2o_stats.cold_start_latency_ms
            print(f"{'Cold Start Latency (ms)':<30} {baseline_avg:<15.2f} {self.v2o_stats.cold_start_latency_ms:<15.2f} {cold_improvement:.1f}×")

        # Warm cache
        if self.v2o_stats.warm_latency_ms:
            warm_improvement = baseline_avg / self.v2o_stats.warm_latency_ms
            print(f"{'Warm Cache Latency (ms)':<30} {'-':<15} {self.v2o_stats.warm_latency_ms:<15.2f} {warm_improvement:.1f}×")

        # Cache hit rate
        print(f"{'Cache Hit Rate (%)':<30} {'0.0':<15} {self.v2o_stats.cache_hit_rate:<15.1f} {'-':<10}")

        print(f"{'-'*70}")
        print(f"\nOverall Speedup: {self.avg_speedup:.1f}× faster with v2o optimizations")

        if self.v2o_stats.speedup_ratio:
            print(f"Cache Benefit: {self.v2o_stats.speedup_ratio:.1f}× faster when cached (warm vs cold)")

        print(f"{'='*70}\n")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "baseline": self.baseline_stats.to_dict(),
            "v2o": self.v2o_stats.to_dict(),
            "overall_speedup": round(self.avg_speedup, 2),
            "cache_benefit": round(self.cache_benefit, 2)
        }


def compare_baseline_vs_v2o(
    model_name: str,
    query: str,
    docs: List[str],
    num_runs: int = 10,
    load_models_func: Optional[callable] = None
) -> ComparisonResult:
    """
    Compare baseline and v2o versions of a reranker.

    Args:
        model_name: Base model name (e.g., "HNSW", "Jina Reranker v2", "BGE-M3")
        query: Query text
        docs: List of documents
        num_runs: Number of test runs (includes cache warmup)
        load_models_func: Optional custom model loading function

    Returns:
        ComparisonResult with detailed statistics
    """
    from utils.models import load_all_models, run_reranker

    if load_models_func is None:
        load_models_func = load_all_models

    # Load both versions
    baseline_name = model_name
    v2o_name = f"{model_name}_v2o"

    print(f"\n🔄 Loading models: {baseline_name} and {v2o_name}...")
    models = load_models_func([baseline_name, v2o_name])

    baseline_model = models[baseline_name]
    v2o_model = models[v2o_name]

    # Profile baseline
    print(f"\n⏱️  Profiling {baseline_name} ({num_runs} runs)...")
    baseline_profiler = V2OProfiler(baseline_name)
    for i in range(num_runs):
        baseline_profiler.profile_query(baseline_model, query, docs)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{num_runs} runs...")

    # Profile v2o
    print(f"\n⏱️  Profiling {v2o_name} ({num_runs} runs)...")
    v2o_profiler = V2OProfiler(v2o_name)
    for i in range(num_runs):
        v2o_profiler.profile_query(v2o_model, query, docs)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{num_runs} runs...")

    # Create comparison result
    result = ComparisonResult(
        model_name=model_name,
        baseline_stats=baseline_profiler.stats,
        v2o_stats=v2o_profiler.stats
    )

    return result


class CacheManager:
    """Utility for managing v2o model caches."""

    @staticmethod
    def get_cache_dir() -> Path:
        """Get the v2o cache directory."""
        return Path.home() / '.cache' / 'maniscope'

    @staticmethod
    def clear_all_caches():
        """Clear all v2o persistent caches."""
        cache_dir = CacheManager.get_cache_dir()
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared all caches at {cache_dir}")
        else:
            print(f"ℹ️  No cache directory found at {cache_dir}")

    @staticmethod
    def clear_model_cache(model_name: str):
        """Clear cache for a specific model."""
        cache_dir = CacheManager.get_cache_dir() / model_name.lower().replace(' ', '_')
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✓ Cleared cache for {model_name}")
        else:
            print(f"ℹ️  No cache found for {model_name}")

    @staticmethod
    def get_cache_stats() -> Dict[str, Dict]:
        """Get statistics for all v2o caches."""
        cache_dir = CacheManager.get_cache_dir()
        stats = {}

        if not cache_dir.exists():
            return stats

        for model_dir in cache_dir.iterdir():
            if model_dir.is_dir():
                # Count files and calculate total size
                num_files = 0
                total_size = 0

                for file in model_dir.rglob('*'):
                    if file.is_file():
                        num_files += 1
                        total_size += file.stat().st_size

                stats[model_dir.name] = {
                    "num_files": num_files,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "path": str(model_dir)
                }

        return stats

    @staticmethod
    def print_cache_stats():
        """Print cache statistics to console."""
        stats = CacheManager.get_cache_stats()

        if not stats:
            print("ℹ️  No v2o caches found")
            return

        print(f"\n{'='*60}")
        print("V2O Cache Statistics")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Files':<10} {'Size (MB)':<15}")
        print(f"{'-'*60}")

        total_files = 0
        total_size = 0.0

        for model, info in stats.items():
            print(f"{model:<30} {info['num_files']:<10} {info['total_size_mb']:<15.2f}")
            total_files += info['num_files']
            total_size += info['total_size_mb']

        print(f"{'-'*60}")
        print(f"{'Total':<30} {total_files:<10} {total_size:<15.2f}")
        print(f"{'='*60}\n")


# Example usage function
def run_v2o_benchmark():
    """
    Example benchmark comparing all v2o models.
    Run this to verify v2o optimizations are working correctly.
    """
    from utils.grid_search import load_dataset

    # Load a small test dataset
    print("Loading test dataset...")
    dataset = load_dataset("dataset-aorb-10.json")

    if not dataset or len(dataset) == 0:
        print("❌ Failed to load dataset")
        return

    # Use first query for testing
    test_query = dataset[0]
    query = test_query['query']
    docs = [doc['text'] for doc in test_query['documents'][:10]]  # First 10 docs

    print(f"\nTest Query: {query[:80]}...")
    print(f"Documents: {len(docs)}")

    # Test each reranker pair
    models_to_test = ["HNSW", "Jina Reranker v2", "BGE-M3"]

    results = {}
    for model_name in models_to_test:
        try:
            print(f"\n{'='*70}")
            print(f"Testing: {model_name}")
            print(f"{'='*70}")

            result = compare_baseline_vs_v2o(
                model_name=model_name,
                query=query,
                docs=docs,
                num_runs=10
            )

            result.print_comparison()
            results[model_name] = result

        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")

    # Print summary
    print(f"\n{'='*70}")
    print("Summary: V2O Optimization Results")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Baseline (ms)':<15} {'V2O (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*70}")

    for model_name, result in results.items():
        baseline_avg = result.baseline_stats.avg_latency_ms
        v2o_avg = result.v2o_stats.avg_latency_ms
        speedup = result.avg_speedup

        print(f"{model_name:<25} {baseline_avg:<15.2f} {v2o_avg:<15.2f} {speedup:<10.1f}×")

    print(f"{'='*70}\n")

    # Save results
    output_dir = Path(__file__).parent.parent.parent.parent / "output" / "v2o_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"v2o_benchmark_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "query": query,
            "num_docs": len(docs),
            "num_runs": 10,
            "results": {name: result.to_dict() for name, result in results.items()}
        }, f, indent=2)

    print(f"✓ Saved benchmark results to {output_file}")

    return results


if __name__ == "__main__":
    # Run benchmark when executed directly
    print("🚀 Running V2O Optimization Benchmark...")
    print("This will compare baseline vs v2o versions for all rerankers")
    print("="*70)

    run_v2o_benchmark()

    print("\n📊 Cache Statistics:")
    CacheManager.print_cache_stats()
