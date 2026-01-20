# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-18

### Added
- **Disk-based embedding cache** for improved performance
  - Automatic caching of document embeddings to avoid re-encoding
  - Configurable cache directory (default: `~/projects/embedding_cache/maniscope`)
  - Cache key based on document content hash + model name
  - `cache_dir` and `use_cache` parameters in `ManiscopeEngine.__init__()`
  - Automatic cache directory creation
  - Graceful error handling for cache failures

### Changed
- Updated `fit()` method to check for cached embeddings before encoding
- Enhanced class docstring to mention caching feature
- Updated README with "Advanced Configuration" section explaining caching

### Testing
- Added 4 new unit tests for caching functionality:
  - `test_caching_enabled`: Verify embeddings are cached and loaded
  - `test_caching_disabled`: Verify caching can be disabled
  - `test_cache_invalidation`: Verify different document sets create separate caches
  - `test_cache_directory_creation`: Verify cache directory is auto-created
- Added `examples/caching_demo.py` to demonstrate caching benefits
- All 11 tests passing (7 original + 4 caching)

### Benefits
- Significant speedup when testing different k/alpha parameters on same corpus
- Faster re-runs after code changes
- Reduced computation time for batch benchmarking
- Especially valuable for large document collections

## [1.0.0] - 2026-01-18

### Added
- Initial release of Maniscope package
- `ManiscopeEngine` class for geodesic reranking
- Two-stage retrieval: Telescope (cosine) + Microscope (geodesic)
- Methods:
  - `fit(docs)`: Build k-NN manifold graph from documents
  - `search(query, top_n)`: Main search with hybrid scoring
  - `search_baseline(query, top_n)`: Pure cosine similarity
  - `search_detailed(query, top_n)`: Detailed score breakdown
  - `compare_methods(query, top_n)`: Compare baseline vs maniscope
- Configurable parameters: `k` (neighbors), `alpha` (hybrid weight)
- Support for custom embedding models via `model_name`
- Comprehensive documentation and examples
- Unit test suite (7 tests)
- MIT License
- Production-ready packaging (setup.py + pyproject.toml)

### Features
- Sub-200ms latency on BEIR benchmarks
- 30-85× faster than cross-encoder/LLM rerankers
- 46% lower energy consumption than baseline
- Robust handling of disconnected graph components
- Clean API with method chaining support
- Proper error handling and type hints

[1.1.0]: https://github.com/digital-duck/maniscope/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/digital-duck/maniscope/releases/tag/v1.0.0
