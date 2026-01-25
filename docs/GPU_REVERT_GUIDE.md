# GPU Acceleration Revert Guide

## Overview

After successfully installing PyTorch 2.5.1+cu121 with GTX 1080 Ti support, this guide documents the changes needed to enable GPU acceleration across all models.

## Verified GPU Status

```
✓ PyTorch 2.5.1+cu121
✓ CUDA 12.1
✓ GPU: NVIDIA GeForce GTX 1080 Ti (Compute Capability 6.1)
✓ GPU tensor operations working
```

## Files to Modify

### 1. ui/utils/models.py (10 locations)

**Strategy:** Change `device = 'cpu'` to auto-detect GPU

**Template:**
```python
# OLD (CPU-only):
device = 'cpu'

# NEW (GPU auto-detect):
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Locations:**

1. **Line 76** - `load_bge_m3_v2o()`
   ```python
   # OLD: device = 'cpu'
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **Line 179** - `load_maniscope_reranker()` v2o
   ```python
   # OLD: device='cpu',  # Force CPU (CUDA kernel incompatibility with GTX 1080 Ti)
   device='cuda' if torch.cuda.is_available() else 'cpu',
   ```

3. **Line 192** - `load_maniscope_reranker()` v1
   ```python
   # OLD: device='cpu',  # Force CPU (CUDA kernel incompatibility)
   device='cuda' if torch.cuda.is_available() else 'cpu',
   ```

4. **Line 202** - `load_maniscope_reranker()` v2
   ```python
   # OLD: device='cpu',  # Force CPU (CUDA kernel incompatibility)
   device='cuda' if torch.cuda.is_available() else 'cpu',
   ```

5. **Line 213** - `load_maniscope_reranker()` v3
   ```python
   # OLD: device='cpu',  # v3 is CPU-friendly (can change to 'cuda' if GPU available)
   device='cuda' if torch.cuda.is_available() else 'cpu',
   ```

6. **Line 225** - `load_maniscope_reranker()` v0
   ```python
   # OLD: device='cpu',
   device='cuda' if torch.cuda.is_available() else 'cpu',
   ```

7. **Line 256** - `load_jina_reranker_v2()`
   ```python
   # OLD: device = "cpu"
   device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

8. **Line 312** - `load_jina_reranker_v2_v2o()`
   ```python
   # OLD: device = "cpu"
   device = "cuda" if torch.cuda.is_available() else "cpu"
   torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
   ```

9. **Line 374** - `load_hnsw_reranker()` baseline
   ```python
   # OLD: model = SentenceTransformer(embedding_model, device='cpu')
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = SentenceTransformer(embedding_model, device=device)
   ```

10. **Line 425** - `load_hnsw_reranker_v2o()`
    ```python
    # OLD: device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ```

**Also update fp16 settings** for Jina models:
```python
# For v2o models, enable fp16 when GPU is available
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
```

### 2. ui/pages/1_🔬_Eval_ReRanker.py (1 location)

**Line 61** - `load_baseline_embeddings()`
```python
# OLD:
def load_baseline_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name, device='cpu')

# NEW:
def load_baseline_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer(model_name, device=device)
```

## Expected Performance Improvements

After enabling GPU:

| Model | CPU Latency | GPU Latency | Speedup |
|-------|-------------|-------------|---------|
| Maniscope v2o | 115ms | 0.4-20ms | 6-290× |
| HNSW v2o | ~50ms | 3-10ms | 5-17× |
| Jina Reranker v2 v2o | ~200ms | 40-60ms | 3-5× |
| BGE-M3 v2o | ~150ms | 50-75ms | 2-3× |
| Baseline embeddings | ~30ms | ~10ms | 3× |

## Verification After Changes

Run these tests to verify GPU is being used:

```bash
# Test 1: Load Maniscope v2o and check device
python -c "
from ui.utils.models import load_maniscope_reranker
model = load_maniscope_reranker(version='v2o')
print(f'Maniscope v2o device: {model.device}')
"

# Test 2: Check GPU memory usage
nvidia-smi

# Test 3: Run a query and measure latency
# Should see dramatic speedup in Eval ReRanker page
```

## Rollback Plan

If GPU causes issues, revert by:
```bash
# Restore CPU-only mode
git checkout ui/utils/models.py ui/pages/1_🔬_Eval_ReRanker.py
```

Or manually change back to:
```python
device = 'cpu'
```

## Notes

- GTX 1080 Ti (compute capability 6.1) is fully supported by PyTorch 2.5.1
- GPU memory usage will increase (~500MB-2GB depending on model)
- v2o models are specifically optimized for GPU acceleration
- First inference may be slower (model loading), subsequent calls will be fast
