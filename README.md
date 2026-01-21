# Maniscope: Efficient Neural Reranking via Geodesic Distances

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Maniscope** is a lightweight geometric reranking method that leverages geodesic distances on k-nearest neighbor manifolds for efficient and accurate information retrieval. It combines global cosine similarity (telescope) with local manifold geometry (microscope) to achieve state-of-the-art retrieval quality with sub-200ms latency.

## Key Features

- **🚀 Fast**: 30-85× faster than LLM/cross-encoder rerankers
- **🎯 Accurate**: MRR 0.9888 on BEIR benchmarks (competitive with SOTA)
- **💡 Efficient**: Sub-200ms latency, 46% lower energy consumption than baseline
- **🌍 Sustainable**: Significantly reduced CO₂ emissions vs parameter-heavy models
- **📊 Robust**: Handles disconnected graph components gracefully
- **🔧 Simple**: Clean API, easy integration with existing RAG systems

## Installation

```bash
pip install maniscope
```

Or install from source:

```bash
git clone https://github.com/digital-duck/maniscope.git
cd maniscope
pip install -e .
```

## Quick Start

### Option 1: Interactive Evaluation App (Recommended)

Launch the Streamlit evaluation interface to benchmark and visualize results:

```bash
# Install dependencies
pip install -e .

# Launch the app
python run_app.py
```

The app provides:
- 📊 **Benchmark Suite**: 6 BEIR datasets (AorB, SciFact, MS MARCO, TREC-COVID, ArguAna, FiQA)
- ⚡ **Optimization Comparison**: Test v0, v1, v2, v3, and v2o versions side-by-side
- 📈 **Analytics Dashboard**: MRR, NDCG@K, MAP, latency analysis
- 🎯 **Live Benchmarking**: Real-time comparison with BGE-M3, Jina Reranker, LLM rerankers
- 💾 **Export Results**: Publication-ready tables and figures

### Option 2: Python API

```python
from maniscope import ManiscopeEngine_v2o

# Initialize engine (v2o: Ultimate optimization - 20-235× speedup)
engine = ManiscopeEngine_v2o(
    model_name='all-MiniLM-L6-v2',
    k=5,              # Number of nearest neighbors
    alpha=0.3,        # Hybrid scoring weight
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
| **v1** | GPU + graph caching | 3.0× | GPU available |
| **v2** | FAISS + scipy + vectorization | 5.0× | Cold cache |
| **v3** | Persistent cache + query LRU | 1-10× | Repeated experiments |
| **v2o** | 🌟 **RECOMMENDED** - All optimizations | **20-235×** | Production |

**v2o Performance (Real-World Results):**
- MS MARCO: 132ms → 0.58ms (229× speedup)
- TREC-COVID: 85ms → 0.38ms (226× speedup)
- SciFact: 92ms → 0.39ms (235× speedup)
- 100% accuracy preservation (MRR=1.0)

### Using Optimized Versions

```python
# v2o: Ultimate optimization (recommended)
from maniscope import ManiscopeEngine_v2o
engine = ManiscopeEngine_v2o(
    k=5, alpha=0.3,
    device=None,       # Auto-detect GPU
    use_cache=True,    # Persistent disk cache
    use_faiss=True     # GPU-accelerated k-NN
)

# v3: CPU-friendly with caching
from maniscope import ManiscopeEngine_v3
engine = ManiscopeEngine_v3(k=5, alpha=0.3, use_cache=True)

# v2: Fast cold-cache performance
from maniscope import ManiscopeEngine_v2
engine = ManiscopeEngine_v2(k=5, alpha=0.3, use_faiss=True)

# v1: Simple GPU acceleration
from maniscope import ManiscopeEngine_v1
engine = ManiscopeEngine_v1(k=5, alpha=0.3)

# v0: Baseline
from maniscope import ManiscopeEngine
engine = ManiscopeEngine(k=5, alpha=0.3)
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
    alpha=0.3,
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

The repository includes 6 BEIR benchmark datasets (12 files total: full + quick test versions):

| Dataset | Queries | Domain | Description |
|---------|---------|--------|-------------|
| **AorB** | 50 | Disambiguation | Semantic word sense disambiguation |
| **SciFact** | 100 | Scientific | Scientific claim verification |
| **MS MARCO** | 200 | Web Search | Web search queries |
| **TREC-COVID** | 50 | Medical | COVID-19 research papers |
| **ArguAna** | 100 | Argumentation | Counter-argument retrieval |
| **FiQA** | 100 | Finance | Financial question answering |

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

## Performance

Benchmarked on 6 BEIR datasets (600 queries total):

| Dataset | MRR | Latency | Speedup vs BGE-M3 |
|---------|-----|---------|-------------------|
| MS MARCO | 1.0000 | 81ms | 85× |
| TREC-COVID | 1.0000 | 142ms | 45× |
| ArguAna | 0.9912 | 152ms | 35× |
| SciFact | 0.9821 | 95ms | 68× |
| FiQA | 0.9707 | 108ms | 52× |

**Average: MRR 0.9888, 116ms latency, 30-85× speedup**

## Citation

If you use Maniscope in your research, please cite:

```bibtex
@article{gong2026maniscope,
  title={Maniscope: Efficient Neural Reranking via Geodesic Distances on k-NN Manifolds},
  author={Gong, Wen G. and Gong, Albert},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

---

**"Look closer to see farther"** — The Maniscope philosophy
