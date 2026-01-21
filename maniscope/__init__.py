"""
Maniscope: Efficient Neural Reranking via Geodesic Distances

A lightweight geometric reranking method that leverages geodesic distances
on k-nearest neighbor manifolds for information retrieval and RAG systems.

Available Versions:
- ManiscopeEngine (v0): Baseline implementation
- ManiscopeEngine_v1: GPU + graph caching (3× speedup)
- ManiscopeEngine_v2: FAISS + scipy (5× speedup)
- ManiscopeEngine_v3: Persistent cache (1-10× speedup)
- ManiscopeEngine_v2o: Ultimate optimization (20-235× speedup) - RECOMMENDED

Example:
    >>> from maniscope import ManiscopeEngine_v2o
    >>> engine = ManiscopeEngine_v2o(k=5, alpha=0.3)
    >>> engine.fit(documents)
    >>> results = engine.search("your query", top_n=10)
"""

__version__ = "1.0.0"
__author__ = "Wen G. Gong"
__license__ = "MIT"

# Import all engine versions
from .maniscope_engine import (
    ManiscopeEngine,           # v0: Baseline
    ManiscopeEngine_v1,        # v1: GPU + graph caching (3× speedup)
    ManiscopeEngine_v2,        # v2: FAISS + scipy (5× speedup)
    ManiscopeEngine_v3,        # v3: Persistent cache (1-10× speedup)
    ManiscopeEngine_v2o,       # v2o: Ultimate optimization (20-235× speedup)
    compare_maniscope_performance
)

__all__ = [
    'ManiscopeEngine',
    'ManiscopeEngine_v1',
    'ManiscopeEngine_v2',
    'ManiscopeEngine_v3',
    'ManiscopeEngine_v2o',
    'compare_maniscope_performance'
]
