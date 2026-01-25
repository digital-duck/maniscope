# Troubleshooting Guide

## CUDA/GPU Issues

If you encounter CUDA errors like "unspecified launch failure" or GPU-related problems:

### Quick Fix: Force CPU Mode

Set this environment variable before running Maniscope:

```bash
export MANISCOPE_FORCE_CPU=true
```

Then restart your Streamlit session. This will force all embedding computations to use CPU instead of GPU.

### Alternative: Per-Session Fix

If you can't set environment variables, the system will automatically fall back to CPU when CUDA fails and show a warning message.

## Performance Notes

- **CPU Mode**: Slower but more compatible, works on all systems
- **GPU Mode**: Faster but may have compatibility issues depending on your CUDA setup

## Common CUDA Issues

- **Driver mismatch**: CUDA driver version doesn't match PyTorch CUDA version
- **Memory issues**: GPU running out of memory
- **Multi-process conflicts**: Multiple processes trying to use CUDA simultaneously

The CPU fallback ensures Maniscope works reliably regardless of your GPU setup.