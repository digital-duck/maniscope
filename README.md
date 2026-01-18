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

```python
from maniscope import ManiscopeEngine

# Initialize engine
engine = ManiscopeEngine(
    model_name='all-MiniLM-L6-v2',
    k=5,              # Number of nearest neighbors
    alpha=0.3,        # Hybrid scoring weight
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

## Performance

Benchmarked on 5 BEIR datasets (550 queries total):

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
