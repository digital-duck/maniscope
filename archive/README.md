# Archive Directory

This directory contains archived files from the repository migration and cleanup process. All files are preserved for reference and historical review.

## Contents

### Engine Files

Located in `archive/code/` (3 engine files):

1. **`engine_old_backup.py`** (644 lines)
   - **Original location:** `maniscope/engine.py`
   - **Status:** OLD backup file
   - **Description:** Original ManiscopeEngine implementation (v0 baseline only)
   - **Why archived:** Contains only the baseline version; superseded by `maniscope/maniscope_engine.py` which contains all versions (v0-v2o)
   - **Date archived:** 2026-01-20

2. **`maniscope_engine_ui_duplicate.py`** (1621 lines)
   - **Original location:** `ui/utils/maniscope_engine.py`
   - **Status:** Duplicate copy
   - **Description:** Complete ManiscopeEngine with all optimization versions (v0, v1, v2, v3, v2o)
   - **Why archived:** Exact duplicate of `maniscope/maniscope_engine.py`
   - **Date archived:** 2026-01-20
   - **Action taken:** Updated UI to import from package (`from maniscope import ManiscopeEngine_v2o`) instead of local copy

3. **`maniscope_engine_v0.py`** (9.5K)
   - **Status:** Old version from migration
   - **Description:** Earlier version of ManiscopeEngine
   - **Why archived:** Superseded by current implementation
   - **Date archived:** 2026-01-20

## Active Engine File

The **ONLY** active engine file is:
- **`maniscope/maniscope_engine.py`** (1621 lines)
  - Contains all versions: v0, v1, v2, v3, v2o
  - Exported via `maniscope/__init__.py`
  - Used by both the package and the UI

## Import Usage

**Package imports (recommended):**
```python
from maniscope import ManiscopeEngine_v2o  # Ultimate optimization (20-235× speedup)
from maniscope import ManiscopeEngine_v1   # GPU + graph caching (3× speedup)
from maniscope import ManiscopeEngine_v2   # FAISS + scipy (5× speedup)
from maniscope import ManiscopeEngine_v3   # Persistent cache (1-10× speedup)
from maniscope import ManiscopeEngine      # Baseline (v0)
```

**All UI files now use package imports:**
- `ui/utils/models.py` - Updated to import from `maniscope` package
- `ui/utils/grid_search.py` - Updated to import from `maniscope` package

## Safety Note

These files are archived (not deleted) to:
1. Preserve all code for reference
2. Allow comparison with old implementations if needed
3. Enable recovery if needed during review

The `archive/` directory is included in `.gitignore` so these files won't be committed to version control.

### Documentation Files

Located in `archive/docs/` (8 migration-specific documents):

1. **`CLEANUP_SUMMARY.md`** - Repository cleanup summary (2026-01-20)
2. **`ENGINE_CONSOLIDATION.md`** - Engine file consolidation documentation
3. **`FINAL_REVIEW.md`** - Final review checklist after migration
4. **`MIGRATION_REVIEW.md`** - Detailed migration review and verification
5. **`PROFESSIONAL_RENAME.md`** - Professional naming improvements
6. **`REFACTORING_SUMMARY.md`** - Code refactoring summary
7. **`REFACTORING_UI.md`** - UI refactoring documentation (src/app → ui)
8. **`TEST_RESULTS.md`** - Test results from migration

**Purpose**: These documents detail the migration process from the RAG-ReRanker research repository to the production maniscope package. They are preserved for historical reference but are no longer needed for daily use.

## Archive Structure

```
archive/
├── README.md                           # This file
├── code/                               # Archived engine files (3 files)
│   ├── engine_old_backup.py            # Old v0-only engine (644 lines)
│   ├── maniscope_engine_ui_duplicate.py # Duplicate UI copy (1621 lines)
│   └── maniscope_engine_v0.py          # Earlier version (9.5K)
└── docs/                               # Migration documentation (8 files)
    ├── CLEANUP_SUMMARY.md
    ├── ENGINE_CONSOLIDATION.md
    ├── FINAL_REVIEW.md
    ├── MIGRATION_REVIEW.md
    ├── PROFESSIONAL_RENAME.md
    ├── REFACTORING_SUMMARY.md
    ├── REFACTORING_UI.md
    └── TEST_RESULTS.md
```

---

**Last updated:** 2026-01-20
**By:** Repository cleanup and organization
**Note:** All archived files are preserved but not needed for production use
