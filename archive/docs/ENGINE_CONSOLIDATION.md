# Engine File Consolidation

**Date:** 2026-01-20
**Status:** ✅ Complete

## Problem

The repository had **three** engine files with overlapping functionality:

1. `/maniscope/engine.py` (644 lines) - Old backup
2. `/maniscope/maniscope_engine.py` (1621 lines) - Package version
3. `/ui/utils/maniscope_engine.py` (1621 lines) - UI local copy

This caused:
- Code duplication
- Confusion about which file is active
- Maintenance burden (need to update multiple files)
- Import inconsistency (package vs local imports)

## Solution

### 1. ✅ Identified Active Engine File

**Active file:** `maniscope/maniscope_engine.py` (1621 lines)
- Contains all optimization versions: v0, v1, v2, v3, v2o
- Exported via `maniscope/__init__.py`
- Single source of truth for the package

### 2. ✅ Archived Unused Files

Created `archive/` directory and moved:

| Original Location | New Location | Reason |
|-------------------|--------------|--------|
| `maniscope/engine.py` | `archive/engine_old_backup.py` | Old backup (v0 only) |
| `ui/utils/maniscope_engine.py` | `archive/maniscope_engine_ui_duplicate.py` | Duplicate of package version |

### 3. ✅ Updated UI Imports

**Before:**
```python
# ui/utils/models.py
from maniscope_engine import ManiscopeEngine_v2o  # Local file
```

**After:**
```python
# ui/utils/models.py
from maniscope import ManiscopeEngine_v2o  # Package import
```

**Files updated:**
- `ui/utils/models.py` - All 5 version imports (v0, v1, v2, v3, v2o)
- `ui/utils/grid_search.py` - ManiscopeEngine import

### 4. ✅ Updated .gitignore

**Before:**
```gitignore
# Backup files (not used in package)
maniscope/engine.py
```

**After:**
```gitignore
archive/
```

The archive directory is now gitignored, so archived files won't be committed.

### 5. ✅ Created .env.example

Created `.env.example` with API key templates based on RAG-ReRanker usage:

```bash
# OpenRouter API (for LLM-based reranking)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Ollama (for local LLM-based reranking)
# OLLAMA_URL=http://localhost:11434/v1

# HuggingFace (optional)
# HF_TOKEN=your_huggingface_token_here
```

## Final Structure

```
maniscope/
├── archive/                          # ← NEW: Archived unused files
│   ├── README.md                     # Documentation
│   ├── engine_old_backup.py          # Old v0-only backup
│   └── maniscope_engine_ui_duplicate.py  # Former UI duplicate
├── maniscope/
│   ├── __init__.py                   # Exports all versions
│   └── maniscope_engine.py # ← ACTIVE: Single source of truth
├── ui/
│   └── utils/
│       ├── models.py                 # ✅ Updated: imports from package
│       └── grid_search.py            # ✅ Updated: imports from package
├── .env.example                      # ← NEW: API key template
└── .gitignore                        # ✅ Updated: ignores archive/
```

## Verification

### Package Import Test
```bash
$ python -c "from maniscope import ManiscopeEngine_v2o; print('✅ PASS')"
✅ PASS
```

### UI Import Test
```bash
$ python -c "
import sys
sys.path.insert(0, 'ui/utils')
from models import load_maniscope_reranker
model = load_maniscope_reranker(version='v2o', k=5, alpha=0.3)
print(f'✅ Created {type(model).__name__}')
"
✅ Created ManiscopeEngine_v2o
```

### All Tests
```bash
$ ./scripts/QUICK_TEST.sh
✅ PASS - Package Import
✅ PASS - Engine Functionality
✅ PASS - Directory Structure
✅ PASS - Datasets (12 files)
✅ PASS - Path References
```

## Benefits

1. **Single Source of Truth** - One active engine file (`maniscope_engine.py`)
2. **Consistent Imports** - UI uses package imports (not local copies)
3. **Clean Structure** - Unused files archived (not deleted)
4. **Easy Maintenance** - Update one file, changes propagate everywhere
5. **API Key Template** - `.env.example` documents required environment variables
6. **Version Control** - Archive directory gitignored to avoid committing old code

## Import Guidelines

**✅ Correct (use everywhere):**
```python
from maniscope import ManiscopeEngine_v2o
from maniscope import ManiscopeEngine_v1
from maniscope import ManiscopeEngine_v2
from maniscope import ManiscopeEngine_v3
from maniscope import ManiscopeEngine
```

**❌ Incorrect (don't use):**
```python
from maniscope_engine import ManiscopeEngine_v2o  # Local file (removed)
from ui.utils.maniscope_engine import ...         # Duplicate (archived)
```

## Next Steps

1. **Review archived files** - Check `archive/` and delete if not needed
2. **Test Streamlit app** - Verify UI works with package imports
3. **Document API usage** - Update USAGE.md with .env setup instructions
4. **Consider .env setup** - Add instructions for setting up OpenRouter API key

---

**Performed by:** Repository consolidation and cleanup
**Status:** Ready for testing and review
**All tests:** Passing ✅
