# Maniscope Repository Structure

## Overview

This repository contains the **independent, self-contained Maniscope package** for efficient neural reranking via geodesic distances. It has been refactored from the RAG-ReRanker research codebase to be a clean, production-ready Python package with no dependencies on the separate Semanscope project.

## Directory Structure

```
maniscope/
├── maniscope/              # Core package
│   ├── __init__.py        # Package initialization and exports
│   └── engine.py          # ManiscopeEngine core implementation
├── examples/              # Usage examples
│   └── basic_usage.py    # Basic usage demonstration
├── tests/                 # Unit tests
│   └── test_engine.py    # Engine tests (pytest)
├── docs/                  # Documentation (placeholder)
├── setup.py              # Package installation (legacy)
├── pyproject.toml        # Modern Python packaging config
├── requirements.txt      # Dependencies
├── README.md             # Main documentation
├── LICENSE               # MIT License
└── .gitignore           # Git ignore patterns
```

## Key Files

### Core Implementation

**`maniscope/engine.py`** (450 lines)
- `ManiscopeEngine` class
- Methods:
  - `fit(docs)`: Build k-NN manifold graph
  - `search(query, top_n)`: Main search (telescope + microscope)
  - `search_baseline(query, top_n)`: Baseline cosine similarity
  - `search_detailed(query, top_n)`: Detailed score breakdown
  - `compare_methods(query, top_n)`: Compare baseline vs maniscope

### Package Configuration

**`setup.py`** - Traditional setuptools configuration
**`pyproject.toml`** - Modern PEP 518/621 configuration
**`requirements.txt`** - Core dependencies:
- numpy>=1.21.0
- networkx>=2.6.0
- scikit-learn>=1.0.0
- sentence-transformers>=2.2.0
- torch>=1.10.0

### Documentation

**`README.md`** - Comprehensive package documentation:
- Installation instructions
- Quick start guide
- API reference
- Performance benchmarks
- Environmental impact analysis
- Citation

## Key Differences from RAG-ReRanker

This is a **clean extraction** with:

✅ **No Streamlit dependencies** - Pure Python library
✅ **No ties to st_semantics** - Completely independent
✅ **Production-ready API** - Clean, documented methods
✅ **Proper packaging** - setup.py + pyproject.toml
✅ **Unit tests** - pytest test suite
✅ **MIT License** - Open source ready
✅ **Examples** - Runnable demo scripts

## Installation

### For Development

```bash
cd ~/projects/digital-duck/maniscope
pip install -e .
```

### For Users (when published)

```bash
pip install maniscope
```

## Testing

```bash
cd ~/projects/digital-duck/maniscope
pytest tests/
```

## Running Examples

```bash
cd ~/projects/digital-duck/maniscope
python examples/basic_usage.py
```

## Dependencies

**No overlap with Semanscope:**
- Independent sentence-transformers usage
- Own k-NN graph construction
- No shared utility files

**Core dependencies:**
- sentence-transformers: Text embeddings
- networkx: Graph algorithms (shortest paths)
- scikit-learn: k-NN construction
- numpy: Numerical operations
- torch: Backend for transformers

## Version

**v1.0.0** - Initial release
- Extracted from RAG-ReRanker research code
- Clean, production-ready API
- Ready for PyPI publication
- Accompanies arXiv paper v2.0

## Next Steps

1. ✅ Test package installation: `pip install -e .`
2. ✅ Run unit tests: `pytest tests/`
3. ✅ Run example: `python examples/basic_usage.py`
4. 📦 Create GitHub repository
5. 📝 Add comprehensive docstrings
6. 🚀 Publish to PyPI
7. 📚 Create ReadTheDocs documentation

## Maintenance

This package is maintained independently from:
- `/home/gongai/projects/digital-duck/st_semantics` (Semanscope)
- `/home/gongai/projects/digital-duck/st_semantics/research/RAG-ReRanker` (Research code)

All updates should be made in this repository to keep it self-contained.

## Contact

- **Authors**: Wen G. Gong, Albert Gong
- **Email**: wen.gong.research@gmail.com
- **GitHub**: https://github.com/digital-duck/maniscope (to be created)
- **Paper**: arXiv:XXXX.XXXXX (v2.0)

---

**Status**: ✅ Ready for independent release
**Date Created**: 2026-01-18
**Refactored From**: RAG-ReRanker/src/app/utils/maniscope_engine.py
