#!/usr/bin/env python3
"""ndcg_sweep_M_v2.py — Maniscope NDCG@10 + latency across candidate pool sizes.

Benchmark-consistent protocol (fair comparison across all M values):

  M=10   Uses the pre-built benchmark dataset (data/dataset-nfcorpus.json)
         where relevant documents are guaranteed to be in the candidate set.
         Serves as the reproducibility baseline for the paper's reported value.

  M>10   Extends each query's 10 benchmark candidates with BM25-retrieved
         negatives from the full NFCorpus corpus to reach pool size M.
         Relevant documents remain in the pool, so NDCG@10 values are directly
         comparable across all M values.

Usage (from RAG-ReRanker root):
    python3 scripts/ndcg_sweep_M_v2.py                    # full sweep
    python3 scripts/ndcg_sweep_M_v2.py --m 10             # sanity check
    python3 scripts/ndcg_sweep_M_v2.py --m 50,100,200,500 # skip M=10
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'app' / 'utils'))

from beir.datasets.data_loader import GenericDataLoader
from maniscope_engine import ManiscopeEngine_v2o
from rank_bm25 import BM25Okapi

# ── Paper latency reference (from latency_sweep_M.py, controlled benchmark) ──
PAPER_LATENCY: Dict[int, Tuple[float, float, float]] = {
    10:  (26.23,  20.67,  27.41),
    50:  (70.11,  69.71,  72.57),
    100: (144.27, 144.61, 147.33),
    200: (276.78, 276.59, 278.17),
    500: (680.60, 681.28, 684.14),
}

# ── BM25 helpers ──────────────────────────────────────────────────────────────

def build_bm25_index(corpus: dict) -> Tuple[BM25Okapi, List[str]]:
    """Build a BM25Okapi index over a BEIR corpus dict.

    Args:
        corpus: BEIR corpus {doc_id: {"title": ..., "text": ...}}.

    Returns:
        (bm25, doc_ids) where doc_ids[i] corresponds to BM25 row i.
    """
    doc_ids = list(corpus.keys())
    tokenised = [
        (corpus[d].get('title', '') + ' ' + corpus[d].get('text', ''))
        .strip().lower().split()
        for d in doc_ids
    ]
    return BM25Okapi(tokenised), doc_ids


def bm25_top_k(bm25: BM25Okapi, doc_ids: List[str],
               query: str, top_k: int) -> List[str]:
    """Return the top-k doc_ids for *query* according to BM25."""
    scores = bm25.get_scores(query.lower().split())
    return [doc_ids[i] for i in np.argsort(-scores)[:top_k]]


# ── Evaluation ────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_indices: List[int], relevant_indices: set, k: int = 10) -> float:
    """Compute NDCG@k given a ranked list of candidate indices and a relevance set.

    Args:
        ranked_indices:   Candidate indices in ranked order (best first).
        relevant_indices: Set of indices that are relevant.
        k:                Cutoff rank.

    Returns:
        NDCG@k score in [0, 1].
    """
    dcg  = sum(1.0 / math.log2(r + 2)
               for r, idx in enumerate(ranked_indices[:k]) if idx in relevant_indices)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(relevant_indices), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ── Candidate pool construction ───────────────────────────────────────────────

def build_candidate_pool(
    bench_docs: List[str],
    query: str,
    target_m: int,
    bm25: BM25Okapi,
    bm25_doc_ids: List[str],
    corpus: dict,
) -> List[str]:
    """Return a candidate pool of exactly *target_m* documents.

    The first 10 entries are always the benchmark candidates (relevant docs
    guaranteed present).  For target_m > 10, BM25-retrieved documents from
    the full corpus are appended as additional negatives, skipping any text
    already in the benchmark set to avoid duplicates.

    Args:
        bench_docs:    10-document benchmark candidate list for this query.
        query:         Query text used for BM25 retrieval.
        target_m:      Desired pool size.
        bm25:          Pre-built BM25 index over the full corpus.
        bm25_doc_ids:  Document IDs corresponding to BM25 index rows.
        corpus:        Full BEIR corpus dict for text lookup.

    Returns:
        List of document texts of length min(target_m, available_docs).
    """
    if target_m <= 10:
        return bench_docs[:target_m]

    pool      = bench_docs[:]
    bench_set = set(bench_docs)

    for did in bm25_top_k(bm25, bm25_doc_ids, query, top_k=target_m * 2):
        if len(pool) >= target_m:
            break
        text = (corpus[did].get('title', '') + ' ' + corpus[did].get('text', '')).strip()
        if text not in bench_set:
            pool.append(text)

    return pool[:target_m]


# ── Core sweep logic ──────────────────────────────────────────────────────────

def run_sweep_for_m(
    m: int,
    benchmark: list,
    engine: ManiscopeEngine_v2o,
    bm25: BM25Okapi,
    bm25_doc_ids: List[str],
    corpus: dict,
) -> Dict[str, float]:
    """Run the full 323-query evaluation for a single pool size M.

    Returns:
        Dict with keys ndcg10, lat_mean, lat_p50, lat_p95.
    """
    ndcg_scores: List[float] = []
    latencies:   List[float] = []

    for item in tqdm(benchmark, desc=f"M={m:>4}", leave=False):
        query_text  = item['query']
        rel_indices = {int(k) for k, v in item['relevance_map'].items() if v > 0}

        cand_texts = build_candidate_pool(
            bench_docs=item['docs'],
            query=query_text,
            target_m=m,
            bm25=bm25,
            bm25_doc_ids=bm25_doc_ids,
            corpus=corpus,
        )

        engine.fit(cand_texts)

        t_start = time.perf_counter()
        ranked  = engine.search_maniscope_detailed(query_text, top_n=10)
        latencies.append((time.perf_counter() - t_start) * 1000)

        reranked = [r['doc_id'] for r in ranked if r['doc_id'] < len(cand_texts)]
        ndcg_scores.append(ndcg_at_k(reranked, rel_indices, k=10))

    p50, p95 = np.percentile(latencies, [50, 95])
    return {
        'ndcg10':   float(np.mean(ndcg_scores)),
        'lat_mean': float(np.mean(latencies)),
        'lat_p50':  float(p50),
        'lat_p95':  float(p95),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option('--m', 'm_str', default='10,50,100,200,500', show_default=True,
              help='Comma-delimited candidate pool sizes to evaluate.')
@click.option('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
              show_default=True, help='Sentence-transformers model name.')
@click.option('--bench', default='data/dataset-nfcorpus.json', show_default=True,
              help='Benchmark dataset JSON (M=10 baseline, relevant docs guaranteed).')
@click.option('--corpus', 'corpus_dir', default='beir/datasets/nfcorpus', show_default=True,
              help='BEIR NFCorpus directory for BM25 extension.')
@click.option('--out', default='output/ndcg_sweep_M_nfcorpus_v2.json', show_default=True,
              help='Output JSON path.')
@click.option('--k', default=5, show_default=True, help='Maniscope graph connectivity k.')
@click.option('--alpha', default=0.5, show_default=True, help='Maniscope blend weight α.')
def main(m_str: str, model: str, bench: str, corpus_dir: str,
         out: str, k: int, alpha: float) -> None:
    """Evaluate Maniscope NDCG@10 and latency across candidate pool sizes M."""

    m_values = [int(x.strip()) for x in m_str.split(',')]

    # ── Validate inputs ───────────────────────────────────────────────────────
    for path, label in [(bench, '--bench'), (corpus_dir, '--corpus')]:
        if not Path(path).exists():
            raise click.BadParameter(f"Path not found: {path}", param_hint=label)

    # ── Load data ─────────────────────────────────────────────────────────────
    click.echo(f"Loading benchmark dataset ({bench})...")
    with open(bench) as f:
        benchmark: list = json.load(f)
    click.echo(f"  {len(benchmark)} queries, {len(benchmark[0]['docs'])} docs each")

    click.echo("Loading full NFCorpus for BM25 extension...")
    corpus, _, _ = GenericDataLoader(corpus_dir).load(split='test')
    click.echo(f"  Corpus: {len(corpus)} docs")

    click.echo("Building BM25 index...")
    bm25, bm25_doc_ids = build_bm25_index(corpus)
    click.echo("  Index ready")

    click.echo(f"Loading Maniscope engine ({model})...")
    engine = ManiscopeEngine_v2o(
        model_name=model, k=k, alpha=alpha,
        verbose=False, use_cache=False, use_faiss=False)
    click.echo("  Engine ready\n")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    click.echo(f"Running M sweep over {len(benchmark)} queries...\n")
    all_results: Dict[int, Dict[str, float]] = {}

    for M in m_values:
        t0      = time.perf_counter()
        result  = run_sweep_for_m(M, benchmark, engine, bm25, bm25_doc_ids, corpus)
        elapsed = time.perf_counter() - t0
        all_results[M] = result

        click.echo(
            f"  M={M:>4}: NDCG@10={result['ndcg10']:.4f}  "
            f"lat mean={result['lat_mean']:6.2f}ms  "
            f"p50={result['lat_p50']:6.2f}ms  "
            f"p95={result['lat_p95']:6.2f}ms  "
            f"({len(benchmark)} queries, {elapsed:.1f}s)"
        )

    # ── LaTeX output ──────────────────────────────────────────────────────────
    click.echo("\n" + "=" * 70)
    click.echo("Table 5 rows (for LaTeX) — NDCG@10 | Mean | p50 | p95:")
    click.echo("=" * 70)
    for M in m_values:
        r  = all_results[M]
        pl = PAPER_LATENCY.get(M, (r['lat_mean'], r['lat_p50'], r['lat_p95']))
        click.echo(f"  {M:<5} & {r['ndcg10']:.4f} & {pl[0]:.2f} & {pl[1]:.2f} & {pl[2]:.2f} \\\\")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model':    model,
        'k':        k,
        'alpha':    alpha,
        'protocol': 'benchmark-consistent-v2',
        'results':  all_results,
    }
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    click.echo(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
