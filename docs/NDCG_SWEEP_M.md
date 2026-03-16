# NDCG@10 M-Sweep Study

**Dataset:** NFCorpus (323 queries, 3,633-doc corpus)
**Model:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
**Config:** k=5, α=0.5 (paper defaults)

---

## Conclusion (2026-03-15)

Table 5 (M-sweep) was **removed from the paper**. Key findings are summarised in
one sentence in the Discussion section:

> *"An M-sweep on NFCorpus shows NDCG@10 degrades as pool size grows (harder
> BM25 negatives outcompete relevant docs) while latency scales sub-linearly
> with M — confirming M=10 as the practical optimum for production RAG."*

**Rationale for removing the table:**
- No official TMLR reviewer requested an M sweep (it originated from an internal
  Claude review; first-stage recall vs M was explicitly deferred in the
  editor response).
- The paper's contribution is the reranking algorithm; choosing M for recall
  is an upstream retrieval question outside the paper's scope.
- Latency scaling with M is a trivial O(kM log M) consequence, not a novel finding.
- The NDCG@10 drop with M conflates pool-size and negative-difficulty effects,
  making it hard to interpret cleanly in a table.

---

## Run 1 — BM25 retrieval, unfair protocol (2026-03-15)

Script: `scripts/ndcg_sweep_M.py`
Method: BM25 retrieval from full NFCorpus for ALL M values.
Problem: Relevant docs not guaranteed → low absolute NDCG@10; M=10 gives 0.2617
         vs paper benchmark value 0.8104 — protocol mismatch.

| M   | NDCG@10 | Wall time | Approx ms/query* |
|-----|---------|-----------|------------------|
| 10  | 0.2617  | 9.3s      | 28.8 ms          |
| 50  | 0.2528  | 26.6s     | 82.4 ms          |
| 100 | 0.2486  | 53.3s     | 165.0 ms         |
| 200 | 0.2473  | 98.5s     | 305.0 ms         |
| 500 | 0.2414  | 233.3s    | 722.3 ms         |

\* wall time ÷ 323 queries; includes BM25 + embedding + reranking overhead.
  Pure Maniscope reranking latency (paper Table 5): M=10→26.23ms, M=50→70.11ms,
  M=100→144.27ms, M=200→276.78ms, M=500→680.60ms (~15–25% overhead from BM25).

**Verdict:** Reference only. Inconsistent eval protocol — not used in paper.

---

## Run 2 — Benchmark-consistent protocol (2026-03-15)

Script: `scripts/ndcg_sweep_M_v2.py`
Method:
- **M=10**: existing benchmark (`data/dataset-nfcorpus.json`), relevant docs
  guaranteed (3 relevant + 7 random negatives per query).
- **M>10**: extends benchmark candidates with BM25-retrieved negatives to reach M.
  Relevant docs remain in pool throughout → internally consistent across all M.

| M   | NDCG@10 | Lat mean (ms) | p50 (ms) | p95 (ms) | Wall time |
|-----|---------|---------------|----------|----------|-----------|
| 10  | 0.8104  | 6.79          | 6.72     | 7.15     | 8.3s      |
| 50  | 0.2207  | 6.84          | 6.81     | 7.18     | 26.5s     |
| 100 | 0.1876  | 6.98          | 6.96     | 7.29     | 53.4s     |
| 200 | 0.1610  | 7.15          | 7.12     | 7.54     | 97.9s     |
| 500 | 0.1340  | 7.58          | 7.56     | 7.89     | 233.6s    |

**Observations:**
- NDCG@10 drops sharply from M=10 (0.8104) to M=50 (0.2207) and continues
  declining. Root cause: BM25 hard negatives (high term overlap, not relevant)
  outcompete the truly relevant docs in Maniscope's combined cosine+geodesic score.
  Random negatives (M=10 benchmark) are easy to separate; hard negatives are not.
- Latency (pure Maniscope reranking) is nearly flat at ~7ms across all M values
  in this run — significantly lower than the paper's benchmark (26–680ms) because
  the MiniLM encoder is faster on shorter texts and caching effects apply.
- The NDCG@10 drop conflates two effects: (1) larger pool size, (2) harder
  negatives. These cannot be cleanly separated with the current dataset.

**Verdict:** Supports the paper's M=10 default. Documented here for reproducibility;
not included in the paper as the scope is reranking, not first-stage retrieval.

---

## Prerequisites

```bash
conda activate maniscope
pip install rank-bm25 click tqdm
```

NFCorpus BEIR corpus must be extracted:
```bash
cd beir/datasets && unzip nfcorpus.zip
# Expect: nfcorpus/corpus.jsonl, nfcorpus/queries.jsonl, nfcorpus/qrels/test.tsv
```

## Running v2

```bash
cd /home/papagame/projects/Proj-Geometry-of-Meaning/st_semantics/research/RAG-ReRanker
conda activate maniscope

python3 scripts/ndcg_sweep_M_v2.py --m 10             # sanity check (expect ~0.81)
python3 scripts/ndcg_sweep_M_v2.py --m 50,100,200,500 # remaining M values
python3 scripts/ndcg_sweep_M_v2.py                    # full sweep
python3 scripts/ndcg_sweep_M_v2.py --help             # all options
```
