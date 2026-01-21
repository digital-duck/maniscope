#!/usr/bin/env python3
"""
Performance comparison script for ManiscopeEngine vs ManiscopeEngine__v1
"""

from maniscope import ManiscopeEngine, ManiscopeEngine__v1
import time

# Sample documents
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Python is a non-venomous snake found in tropical regions.",
    "Machine learning frameworks like TensorFlow use Python.",
    "The python snake hunts small mammals and birds.",
    "NumPy and pandas are popular Python libraries for data science.",
    "Ball pythons are popular pets due to their docile nature.",
    "Django is a web framework written in Python.",
    "Pythons are constrictors that squeeze their prey.",
    "Python's syntax emphasizes code readability.",
    "Reticulated pythons are among the longest snakes in the world.",
]

def benchmark_engines():
    print("=" * 80)
    print("ManiscopeEngine Performance Comparison")
    print("=" * 80)

    # Initialize engines
    print("\nInitializing engines...")
    engine = ManiscopeEngine(k=5, alpha=0.3, verbose=False)
    engine_v1 = ManiscopeEngine__v1(k=5, alpha=0.3, verbose=False)

    # Fit engines
    print("Fitting engines on document corpus...")
    engine.fit(documents)
    engine_v1.fit(documents)

    # Test queries - same query repeated to show caching benefit
    repeated_query = "What does a python eat?"
    queries = [repeated_query] * 10  # 10 identical queries

    print(f"\nBenchmarking {len(queries)} queries (with repeats for caching)...")

    # Benchmark original engine
    print("\nBenchmarking original engine...")
    start_time = time.time()
    results_orig = []
    for query in queries:
        results_orig.append(engine.search(query, top_n=5))
    time_orig = time.time() - start_time

    # Benchmark optimized engine
    print("Benchmarking optimized engine...")
    start_time = time.time()
    results_v1 = []
    for i, query in enumerate(queries):
        results_v1.append(engine_v1.search(query, top_n=5))
        if i == len(queries) - 1:  # After last query
            print(f"  Cache size during benchmark: {len(engine_v1.query_cache)}")
    time_v1 = time.time() - start_time

    # Check consistency
    consistent = all(r1 == r2 for r1, r2 in zip(results_orig, results_v1))

    # Results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Original engine time:  {time_orig:.4f}s")
    print(f"Optimized engine time: {time_v1:.4f}s")
    print(f"Speedup: {time_orig/time_v1:.2f}x")
    print(f"Results consistent: {consistent}")
    print(f"Query cache size: {len(engine_v1.query_cache)} (out of {engine_v1.query_cache_size})")
    print(f"Cache keys: {list(engine_v1.query_cache.keys()) if engine_v1.query_cache else 'None'}")

    # Show cache hits (check after all searches are done)
    print(f"\nCache analysis (after all queries):")
    unique_queries = set(queries)
    for query in unique_queries:
        cached = engine_v1._get_cached_query_embedding(query) is not None
        count = queries.count(query)
        print(f"  '{query[:30]}...' -> {'CACHED' if cached else 'NEW'} (used {count} times)")

    return {
        'time_orig': time_orig,
        'time_v1': time_v1,
        'speedup': time_orig/time_v1,
        'consistent': consistent,
        'cache_size': len(engine_v1.query_cache)
    }

if __name__ == "__main__":
    benchmark_engines()