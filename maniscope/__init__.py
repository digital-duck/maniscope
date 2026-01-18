"""
Maniscope: Efficient Neural Reranking via Geodesic Distances

A lightweight geometric reranking method that leverages geodesic distances
on k-nearest neighbor manifolds for information retrieval and RAG systems.

Example:
    >>> from maniscope import ManiscopeEngine
    >>> engine = ManiscopeEngine(k=5, alpha=0.3)
    >>> engine.fit(documents)
    >>> results = engine.search("your query", top_n=10)
"""

__version__ = "1.0.0"
__author__ = "Wen G. Gong, Albert Gong"
__license__ = "MIT"

from .engine import ManiscopeEngine

__all__ = ['ManiscopeEngine']
