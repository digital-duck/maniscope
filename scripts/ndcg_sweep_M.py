#!/usr/bin/env python3
"""Compute Maniscope NDCG@10 across candidate pool sizes M on NFCorpus.

Uses BM25 retrieval (via rank_bm25) to retrieve top-M candidates from the
full NFCorpus corpus for each query, then runs Maniscope reranker and
evaluates NDCG@10.

Run from RAG-ReRanker root:
    python3 scripts/ndcg_sweep_M.py

Output: prints Table 5 NDCG@10 column values for M = 10, 50, 100, 200, 500
"""
import sys, json, math, time
import numpy as np
sys.path.insert(0, 'src/app/utils')

from beir.datasets.data_loader import GenericDataLoader
from maniscope_engine import ManiscopeEngine_v2o

# ── BM25 retrieval ────────────────────────────────────────────────────────────

def build_bm25_index(corpus):
    """Build BM25 index over corpus dict {doc_id: {title, text}}."""
    from rank_bm25 import BM25Okapi
    doc_ids = list(corpus.keys())
    doc_texts = []
    for did in doc_ids:
        doc = corpus[did]
        text = (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
        doc_texts.append(text.lower().split())
    bm25 = BM25Okapi(doc_texts)
    return bm25, doc_ids

def bm25_retrieve(bm25, doc_ids, query_text, top_k):
    """Return top_k (doc_id, text) pairs via BM25."""
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(-scores)[:top_k]
    return [(doc_ids[i], scores[i]) for i in top_idx]

# ── NDCG@k ────────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_doc_ids, relevant_doc_ids, k=10):
    """Compute NDCG@k given ranked list and set of relevant doc_ids."""
    rel_set = set(relevant_doc_ids)
    dcg = 0.0
    for rank, did in enumerate(ranked_doc_ids[:k], start=1):
        if did in rel_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
M_VALUES = [10, 50, 100, 200, 500]

print("Loading NFCorpus...")
corpus, queries, qrels = GenericDataLoader('beir/datasets/nfcorpus').load(split='test')
print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")

print("Building BM25 index...")
bm25, doc_ids = build_bm25_index(corpus)
print(f"  Index ready ({len(doc_ids)} docs)")

print(f"Loading Maniscope engine ({MODEL_NAME})...")
engine = ManiscopeEngine_v2o(
    model_name=MODEL_NAME,
    k=5, alpha=0.5, verbose=False, use_cache=False, use_faiss=False)
print("  Engine ready")

query_ids = list(qrels.keys())
print(f"\nRunning M sweep over {len(query_ids)} queries...\n")

results = {}
for M in M_VALUES:
    ndcg_scores = []
    t0 = time.perf_counter()
    for qi, qid in enumerate(query_ids):
        query_text = queries[qid]
        # Relevant docs for this query
        rel_docs = {did for did, score in qrels[qid].items() if score > 0}
        if not rel_docs:
            continue

        # BM25 retrieval: get top-M candidates
        retrieved = bm25_retrieve(bm25, doc_ids, query_text, M)
        cand_ids   = [did for did, _ in retrieved]
        cand_texts = [(corpus[did].get('title', '') + ' ' + corpus[did].get('text', '')).strip()
                      for did in cand_ids]

        if len(cand_texts) < 2:
            continue

        # Maniscope rerank — returns list of dicts with 'doc_id' (index into cand_texts)
        engine.fit(cand_texts)
        results_list = engine.search_maniscope_detailed(query_text, top_n=min(10, M))
        reranked_ids = [cand_ids[r['doc_id']] for r in results_list if r['doc_id'] < len(cand_ids)]

        score = ndcg_at_k(reranked_ids, rel_docs, k=10)
        ndcg_scores.append(score)

    elapsed = time.perf_counter() - t0
    mean_ndcg = np.mean(ndcg_scores)
    results[M] = mean_ndcg
    print(f"  M={M:>4}: NDCG@10 = {mean_ndcg:.4f}  ({len(ndcg_scores)} queries, {elapsed:.1f}s)")

print("\n" + "="*50)
print("Table 5 NDCG@10 values (for LaTeX):")
print("="*50)
for M in M_VALUES:
    print(f"  {M:<5} & {results[M]:.4f} \\\\")

# Save to JSON
out = {'model': MODEL_NAME, 'k': 5, 'alpha': 0.5, 'results': results}
with open('output/ndcg_sweep_M_nfcorpus.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved: output/ndcg_sweep_M_nfcorpus.json")
