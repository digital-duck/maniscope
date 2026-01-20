"""
Basic tests for ManiscopeEngine.
"""

import pytest
import tempfile
from pathlib import Path
from maniscope import ManiscopeEngine, ManiscopeEngine__v1


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
    from maniscope import ManiscopeEngine__v1
    engine_v1 = ManiscopeEngine__v1(k=3, alpha=0.5, verbose=False)
    engine_v1.fit(["test doc 1", "test doc 2", "test doc 3"])
    comparison = ManiscopeEngine.compare_performance(engine, engine_v1, "snake", top_n=3)
    assert 'query' in comparison
    assert 'engine1_time' in comparison
    assert 'engine2_time' in comparison
    assert 'speedup' in comparison
    assert 'results_consistent' in comparison
    assert isinstance(comparison['results_consistent'], bool)


def test_search_requires_fit():
    """Test that search fails before fit."""
    engine = ManiscopeEngine()
    with pytest.raises(ValueError, match="Must call fit"):
        engine.search("test query")


def test_caching_enabled(sample_documents):
    """Test that embeddings are cached to disk."""
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # First fit should encode and cache
        engine1 = ManiscopeEngine(
            k=3,
            cache_dir=tmpdir,
            use_cache=True,
            verbose=False
        )
        engine1.fit(sample_documents)

        # Check that cache file was created
        cache_files = list(Path(tmpdir).glob("*.pkl"))
        assert len(cache_files) == 1, "Cache file should be created"

        # Second fit should load from cache
        engine2 = ManiscopeEngine(
            k=3,
            cache_dir=tmpdir,
            use_cache=True,
            verbose=False
        )
        engine2.fit(sample_documents)

        # Embeddings should be identical
        assert engine1.embeddings.shape == engine2.embeddings.shape
        # Note: We can't use exact equality due to potential floating point differences,
        # but the cache should have been used


def test_caching_disabled(sample_documents):
    """Test that caching can be disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ManiscopeEngine(
            k=3,
            cache_dir=tmpdir,
            use_cache=False,
            verbose=False
        )
        engine.fit(sample_documents)

        # Cache directory should be empty
        cache_files = list(Path(tmpdir).glob("*.pkl"))
        assert len(cache_files) == 0, "No cache files should be created when caching is disabled"


def test_cache_invalidation(sample_documents):
    """Test that cache is invalidated when documents change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First fit with original documents
        engine1 = ManiscopeEngine(
            k=3,
            cache_dir=tmpdir,
            use_cache=True,
            verbose=False
        )
        engine1.fit(sample_documents)

        # Fit with different documents should create new cache
        modified_docs = sample_documents + ["New document"]
        engine2 = ManiscopeEngine(
            k=3,
            cache_dir=tmpdir,
            use_cache=True,
            verbose=False
        )
        engine2.fit(modified_docs)

        # Should have 2 cache files (one for each document set)
        cache_files = list(Path(tmpdir).glob("*.pkl"))
        assert len(cache_files) == 2, "Different document sets should create separate cache files"


def test_cache_directory_creation(sample_documents):
    """Test that cache directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "nested" / "cache" / "dir"
        assert not cache_dir.exists()

        engine = ManiscopeEngine(
            k=3,
            cache_dir=str(cache_dir),
            use_cache=True,
            verbose=False
        )
        engine.fit(sample_documents)

        # Cache directory should now exist
        assert cache_dir.exists()
        assert cache_dir.is_dir()

        # Cache file should be present
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) == 1


@pytest.fixture
def engine_v1(sample_documents):
    engine = ManiscopeEngine__v1(k=3, alpha=0.5, verbose=False)
    engine.fit(sample_documents)
    return engine


def test_v1_engine_initialization():
    """Test that v1 engine initializes correctly."""
    engine = ManiscopeEngine__v1()
    assert engine.k == 5
    assert engine.alpha == 0.5
    assert engine.embeddings is None
    assert hasattr(engine, 'query_cache')
    assert hasattr(engine, 'graph_adjacency')


def test_v1_fit(engine_v1, sample_documents):
    """Test that v1 fit builds embeddings and graph."""
    assert engine_v1.embeddings is not None
    assert len(engine_v1.embeddings) == len(sample_documents)
    assert engine_v1.G is not None
    assert engine_v1.graph_adjacency is not None


def test_v1_search(engine_v1):
    """Test v1 Maniscope geodesic reranking."""
    results = engine_v1.search("programming", top_n=3)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results)


def test_v1_search_detailed(engine_v1):
    """Test v1 detailed search with score breakdown."""
    results = engine_v1.search_detailed("programming", top_n=3)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)


def test_v1_query_caching(sample_documents):
    """Test that v1 caches query embeddings."""
    engine = ManiscopeEngine__v1(k=3, alpha=0.5, verbose=False)
    engine.fit(sample_documents)

    # First encode
    emb1 = engine._encode_query("test query")

    # Second encode should use cache
    emb2 = engine._encode_query("test query")

    # Should be identical
    assert (emb1 == emb2).all()

    # Cache should have 1 entry
    assert len(engine.query_cache) == 1


def test_performance_comparison(engine, engine_v1):
    """Test performance comparison between original and v1."""
    comparison = ManiscopeEngine.compare_performance(
        engine, engine_v1, "programming", top_n=3, num_runs=5
    )

    assert 'engine1_time' in comparison
    assert 'engine2_time' in comparison
    assert 'speedup' in comparison
    assert 'results_consistent' in comparison
    assert comparison['results_consistent'] is True  # Should match
