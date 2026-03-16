#!/usr/bin/env python3
"""Benchmark Maniscope latency vs candidate pool size M.

Runs 20 trials per M value (10 for M > 100) and reports mean, p50, p95 latency.
Use the output for Table 3 (latency column).

Run from RAG-ReRanker root: python3 scripts/latency_sweep_M.py
"""
import sys, time
import numpy as np
sys.path.insert(0, 'src/app/utils')

from maniscope_engine import ManiscopeEngine_v2o

model = ManiscopeEngine_v2o(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    k=5, alpha=0.5, verbose=False, use_cache=False, use_faiss=False)

query = "What are the effects of climate change on ocean ecosystems?"

print(f"{'M':>6}  {'Mean (ms)':>12}  {'p50 (ms)':>10}  {'p95 (ms)':>10}")
print("-" * 46)
for M in [10, 50, 100, 200, 500, 1000]:
    docs = [f"Document {i}: " + " ".join(f"word{j}" for j in range(100)) for i in range(M)]
    latencies = []
    trials = 20 if M <= 100 else 10
    for _ in range(trials):
        t0 = time.perf_counter()
        model.fit(docs)
        model.search_maniscope_detailed(query, top_n=min(10, M), coarse_multiplier=3)
        latencies.append((time.perf_counter() - t0) * 1000)
    p = np.percentile(latencies, [50, 95])
    print(f"{M:>6}  {np.mean(latencies):>12.2f}  {p[0]:>10.2f}  {p[1]:>10.2f}")
