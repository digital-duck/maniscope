"""
Basic tests for ManiscopeEngine.
"""

import pytest
from maniscope import ManiscopeEngine


@pytest.fixture
def sample_documents():
    return [
        "Python is a programming language",
        "Python is a type of snake",
        "Machine learning uses Python",
        "Snakes are reptiles",
        "Programming requires logic",
    ]


@pytest.fixture
def engine(sample_documents):
    engine = ManiscopeEngine(k=3, alpha=0.5, verbose=False)
    engine.fit(sample_documents)
    return engine


def test_engine_initialization():
    """Test that engine initializes correctly."""
    engine = ManiscopeEngine()
    assert engine.k == 5
    assert engine.alpha == 0.5
    assert engine.embeddings is None


def test_fit(engine, sample_documents):
    """Test that fit builds embeddings and graph."""
    assert engine.embeddings is not None
    assert len(engine.embeddings) == len(sample_documents)
    assert engine.G is not None
    assert engine.G.number_of_nodes() == len(sample_documents)


def test_search_baseline(engine):
    """Test baseline cosine similarity search."""
    results = engine.search_baseline("programming", top_n=3)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
    doc, score, idx = results[0]
    assert isinstance(doc, str)
    assert isinstance(score, float)
    assert isinstance(idx, int)


def test_search(engine):
    """Test Maniscope geodesic reranking."""
    results = engine.search("programming", top_n=3)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results)


def test_search_detailed(engine):
    """Test detailed search with score breakdown."""
    results = engine.search_detailed("programming", top_n=3)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
    
    result = results[0]
    assert 'doc_id' in result
    assert 'document' in result
    assert 'final_score' in result
    assert 'cosine_score' in result
    assert 'geo_score' in result
    assert 'connected' in result


def test_compare_methods(engine):
    """Test method comparison."""
    comparison = engine.compare_methods("snake", top_n=3)
    assert 'query' in comparison
    assert 'baseline_results' in comparison
    assert 'maniscope_results' in comparison
    assert 'ranking_changed' in comparison
    assert isinstance(comparison['ranking_changed'], bool)


def test_search_requires_fit():
    """Test that search fails before fit."""
    engine = ManiscopeEngine()
    with pytest.raises(ValueError, match="Must call fit"):
        engine.search("test query")
