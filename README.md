# Maniscope: A Novel RAG Reranker via Geodesic Distances on k-NN Manifolds

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Maniscope** is a lightweight geometric reranking method that leverages geodesic distances on k-nearest neighbor manifolds for efficient and accurate information retrieval. It combines global cosine similarity (telescope) with local manifold geometry (microscope) to achieve state-of-the-art retrieval quality with sub-5ms latency.

## Key Features

- **🚀 Fast**: 3.2× faster than HNSW, 10-45× faster than cross-encoder rerankers
- **🎯 Accurate**: MRR 0.9642 on 8 BEIR benchmarks (1,233 queries), within 2% of best cross-encoder
- **💡 Efficient**: 4.7ms average latency, outperforms HNSW on hardest datasets (NFCorpus: +7.0%, TREC-COVID: +1.6%, AorB: +2.8% NDCG@3)
- **🌍 Practical**: Achieves near-theoretical-maximum accuracy (within 1.8% of LLM-Reranker) at 420× faster speed
- **📊 Robust**: Handles disconnected graph components gracefully via hybrid scoring
- **🔧 Simple**: Clean API, easy integration with existing RAG systems

## Installation

```bash
conda create -n maniscope python=3.11 -y
conda activate maniscope  
```

install from source:

```bash
git clone https://github.com/digital-duck/maniscope.git
cd maniscope
pip install -e .
```

```bash
pip install maniscope
```

## Quick Start

### Option 1: Interactive Evaluation App (Recommended)

Launch the Streamlit evaluation interface to benchmark and visualize results:

```bash

# Launch the app
streamlit run ui/Maniscope.py

# or
python run_app.py
```

The app provides:
- 📊 **Benchmark Suite**: 8 BEIR datasets (NFCorpus, TREC-COVID, SciFact, FiQA, MS MARCO, ArguAna, FEVER, AorB)
- ⚡ **Optimization Comparison**: Test v0, v1, v2, v3, and v2o versions side-by-side
- 📈 **Analytics Dashboard**: MRR, NDCG@K, MAP, latency analysis
- 🎯 **Live Benchmarking**: Real-time comparison with HNSW, BGE-M3, Jina Reranker v2, LLM rerankers
- 💾 **Export Results**: Publication-ready tables and figures

### Option 2: Python API

```python
from maniscope import ManiscopeEngine_v2o

# Initialize engine (v2o: Ultimate optimization - 13.2× speedup)
engine = ManiscopeEngine_v2o(
    model_name='all-MiniLM-L6-v2',
    k=5,              # Number of nearest neighbors
    alpha=0.5,        # Hybrid scoring weight
    device=None,      # Auto-detect GPU
    use_cache=True,   # Enable persistent disk cache
    verbose=True
)

# Fit on document corpus
documents = [
    "Python is a programming language",
    "Python is a type of snake",
    "Machine learning uses Python",
    # ... more documents
]
engine.fit(documents)

# Search with Maniscope (telescope + microscope)
results = engine.search("What is Python?", top_n=5)
for doc, score, idx in results:
    print(f"[{score:.3f}] {doc}")

# Compare with baseline cosine similarity
comparison = engine.compare_methods("What is Python?", top_n=5)
print(f"Ranking changed: {comparison['ranking_changed']}")
```

## Optimization Versions

Maniscope provides multiple optimization levels for different use cases:

| Version | Description | Speedup | Best For |
|---------|-------------|---------|----------|
| **v0** | Baseline (CPU, no cache) | 1.0× | Reference |
| **v1** | Efficient k-NN construction | 17.8× | Early optimization |
| **v2** | Heap-based Dijkstra | 22.0× | Reduced overhead |
| **v2o** | 🌟 **RECOMMENDED** - SciPy optimized | **13.2×** | Production |
| **v3** | Persistent cache + query LRU | Variable | Repeated experiments |

**v2o Performance (Real-World Results on 8 BEIR datasets, 1,233 queries):**
- Average latency: 4.7ms (3.2× faster than HNSW at 14.8ms)
- Outperforms HNSW on hardest datasets (NFCorpus, TREC-COVID, AorB)
- 10-45× faster than cross-encoder rerankers
- Within 2% of best cross-encoder accuracy (Jina v2)

### Using Optimized Versions

```python
# v2o: Ultimate optimization (recommended)
from maniscope import ManiscopeEngine_v2o
engine = ManiscopeEngine_v2o(
    k=5, alpha=0.5,
    device=None,       # Auto-detect GPU
    use_cache=True,    # Persistent disk cache
    use_faiss=True     # GPU-accelerated k-NN
)

# v3: CPU-friendly with caching
from maniscope import ManiscopeEngine_v3
engine = ManiscopeEngine_v3(k=5, alpha=0.5, use_cache=True)

# v2: Fast cold-cache performance
from maniscope import ManiscopeEngine_v2
engine = ManiscopeEngine_v2(k=5, alpha=0.5, use_faiss=True)

# v1: Simple GPU acceleration
from maniscope import ManiscopeEngine_v1
engine = ManiscopeEngine_v1(k=5, alpha=0.5)

# v0: Baseline
from maniscope import ManiscopeEngine
engine = ManiscopeEngine(k=5, alpha=0.5)
```

## Advanced Configuration

### Embedding Cache

Maniscope automatically caches document embeddings to disk to avoid recomputation. This is especially valuable when:
- Testing different `k` and `alpha` parameters on the same corpus
- Re-running experiments after code changes
- Benchmarking multiple rerankers on the same dataset

```python
engine = ManiscopeEngine_v2o(
    model_name='all-MiniLM-L6-v2',
    k=5,
    alpha=0.5,
    cache_dir='~/projects/embedding_cache/maniscope',  # Custom cache location
    use_cache=True,           # Enable persistent disk cache
    query_cache_size=100      # LRU cache for 100 queries
)
```

**Cache behavior:**
- Cache files are stored in `cache_dir` (default: `~/projects/embedding_cache/maniscope`)
- Cache key is computed from document content + model name
- Embeddings are automatically loaded from cache if available
- Query LRU cache stores recent query embeddings in memory

**Benefits:**
- Avoid expensive re-encoding when testing different parameters
- Faster iteration during development
- Reduced computation time for batch benchmarking
- Query cache provides instant response for repeated queries

## How It Works

Maniscope uses a two-stage retrieval architecture:

### 1. **Telescope** (Global Retrieval)
Broad retrieval using cosine similarity to get top candidates

### 2. **Microscope** (Local Refinement)
Geodesic reranking on k-NN manifold graph:
- Build k-nearest neighbor graph from document embeddings
- Compute geodesic distances on this manifold
- Hybrid scoring: `α × cosine + (1-α) × geodesic`

**Key Insight**: Local manifold structure captures semantic relationships better than global Euclidean distances.

## Datasets

The repository includes 8 BEIR benchmark datasets (1,233 queries total):

| Dataset | Queries | Domain | Description |
|---------|---------|--------|-------------|
| **NFCorpus** | 323 | Medical | Medical/nutrition information retrieval |
| **TREC-COVID** | 50 | Biomedical | COVID-19 research papers |
| **SciFact** | 100 | Scientific | Scientific claim verification |
| **FiQA** | 100 | Financial | Financial question answering |
| **MS MARCO** | 200 | Web Search | Web search queries |
| **ArguAna** | 100 | Argumentation | Counter-argument retrieval |
| **FEVER** | 200 | Fact Checking | Evidence-based claim verification |
| **AorB** | 50 | Disambiguation | Semantic word sense disambiguation |

Each dataset includes:
- `dataset-{name}.json`: Full benchmark dataset
- `dataset-{name}-10.json`: Quick test version (10 queries)

**Dataset Format:**
```json
{
  "corpus": {
    "doc_id": {"text": "document text", "title": "..."}
  },
  "queries": {
    "query_id": "query text"
  },
  "qrels": {
    "query_id": {"doc_id": 1}
  }
}
```

See `data/` directory for all datasets.

## Cleanup (after evaluation)

```bash
conda env remove -n maniscope  
```

## Performance

Benchmarked on 8 BEIR datasets (1,233 queries total):

| Dataset | Queries | MRR | NDCG@3 | Latency | Speedup vs HNSW |
|---------|---------|-----|--------|---------|-----------------|
| **NFCorpus** | 323 | 0.8247 | **0.7063** | 4.6ms | 3.7× |
| **TREC-COVID** | 50 | 1.0000 | **0.9659** | 4.5ms | 3.9× |
| **AorB** | 50 | 0.9483 | **0.8698** | 4.4ms | 1.4× |
| **SciFact** | 100 | 0.9708 | 0.9739 | 4.6ms | 3.8× |
| **ArguAna** | 100 | 0.9912 | 0.9900 | 5.4ms | 3.3× |
| **FiQA** | 100 | 0.9814 | 0.9795 | 4.5ms | 3.7× |
| **MS MARCO** | 200 | 1.0000 | 1.0000 | 4.6ms | 2.4× |
| **FEVER** | 200 | 0.9975 | 0.9978 | 4.7ms | 3.1× |

**Average: MRR 0.9642, NDCG@3 0.9354, 4.7ms latency, 3.2× faster than HNSW**

**Key Results:**
- Outperforms HNSW on hardest datasets: NFCorpus (+7.0%), TREC-COVID (+1.6%), AorB (+2.8% NDCG@3)
- Within 2% of best cross-encoder (Jina v2) while being 10× faster
- LLM-Reranker provides only +1.8% NDCG@3 improvement at 420× latency penalty

## Citation

If you use Maniscope in your research, please cite:

```bibtex
@inproceedings{gong2026maniscope,
  title={A Novel RAG Reranker via Geodesic Distances on k-NN Manifolds},
  author={Gong, Wen G.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

---

**"Look closer to see farther"** — The Maniscope philosophy
