# UI Structure Refactoring Summary

## What Changed

The repository structure has been simplified by moving the Streamlit UI from `src/app/` to `ui/` at the root level.

### Before (Old Structure)
```
maniscope/
├── src/
│   └── app/              # Nested UI location
│       ├── RAG-ReRanker-Eval.py
│       ├── config.py
│       ├── pages/
│       └── utils/
```

### After (New Structure)
```
maniscope/
├── ui/                   # UI at root level
│   ├── RAG-ReRanker-Eval.py
│   ├── config.py
│   ├── pages/
│   └── utils/
```

## Files Modified

### 1. Launch Script
- **File:** `run_app.py`
- **Change:** Updated path from `src/app/RAG-ReRanker-Eval.py` → `ui/RAG-ReRanker-Eval.py`

### 2. UI Pages (7 files)
All page files updated to fix data/output directory paths:
- `ui/pages/1_🔬_Eval_ReRanker.py`
- `ui/pages/2_🎯_Benchmark.py`
- `ui/pages/3_📈_Analytics.py`
- `ui/pages/4_🚀_Batch_Benchmark.py`
- `ui/pages/6_⚙️_Configuration.py`
- `ui/pages/7_⚡_Optimization.py`
- `ui/pages/9_📁_Data_Manager.py`

**Change:** Path references updated from:
```python
# Before
Path(__file__).parent.parent.parent.parent / "data"

# After
Path(__file__).parent.parent.parent / "data"
```

### 3. UI Utils
- **File:** `ui/utils/grid_search.py`
- **Change:** Updated data directory path (removed one `.parent` level)

### 4. Documentation
- **File:** `USAGE.md`
- **Changes:**
  - `streamlit run src/app/RAG-ReRanker-Eval.py` → `streamlit run ui/RAG-ReRanker-Eval.py`
  - `from src.app.utils.metrics` → `from ui.utils.metrics`

### 5. Directory Cleanup
- **Removed:** Empty `src/` directory

## Updated Commands

### Launch App
```bash
# Old
streamlit run src/app/RAG-ReRanker-Eval.py

# New
streamlit run ui/RAG-ReRanker-Eval.py

# Or use the launcher (unchanged)
python run_app.py
```

### Import Path (Python API)
```python
# Old
from src.app.utils.metrics import calculate_metrics

# New
from ui.utils.metrics import calculate_metrics
```

## Benefits

1. **Simpler structure** - UI is directly at root level
2. **Cleaner paths** - Fewer nested directories
3. **Better organization** - Clear separation: `maniscope/` (core), `ui/` (interface), `data/` (datasets)
4. **Easier navigation** - No need to traverse `src/app/`

## Migration Verification

✅ All 7 UI pages updated
✅ All utils files updated
✅ Launch script updated
✅ Documentation updated
✅ No broken path references (verified with grep)
✅ Empty `src/` directory removed

## Testing

To verify the refactoring works:

```bash
# Test launch script
python run_app.py

# Test direct streamlit command
streamlit run ui/RAG-ReRanker-Eval.py

# Test Python imports
python -c "from ui.utils.metrics import calculate_metrics; print('✅ Imports work')"
```

All tests should pass without errors.
