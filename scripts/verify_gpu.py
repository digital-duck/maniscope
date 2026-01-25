#!/usr/bin/env python
"""
Quick verification script to check GPU acceleration is enabled.

Run this after enabling GPU to verify:
    python verify_gpu.py
"""

import sys
print("=" * 70)
print("GPU ACCELERATION VERIFICATION")
print("=" * 70)

# 1. Check PyTorch
print("\n[1/4] Checking PyTorch installation...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"✓ Compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("✗ CUDA not available - GPU acceleration will not work!")
        sys.exit(1)
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

# 2. Test GPU tensor operations
print("\n[2/4] Testing GPU tensor operations...")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✓ Matrix multiplication successful on GPU")
    print(f"✓ Result shape: {z.shape}, Device: {z.device}")
except Exception as e:
    print(f"✗ GPU tensor operations failed: {e}")
    sys.exit(1)

# 3. Check model device settings
print("\n[3/4] Checking model device auto-detection...")
try:
    from ui.utils.models import load_maniscope_reranker
    model = load_maniscope_reranker(version='v2o', k=5, alpha=0.3)
    device = model.device
    print(f"✓ Maniscope v2o device: {device}")
    if 'cuda' in str(device):
        print(f"✓ GPU acceleration ENABLED for Maniscope v2o!")
    else:
        print(f"⚠ Warning: Maniscope v2o using CPU (expected: cuda)")
except Exception as e:
    print(f"⚠ Could not load Maniscope v2o: {e}")
    print("  (This is OK if models not downloaded yet)")

# 4. Check baseline embeddings
print("\n[4/4] Checking baseline embeddings...")
try:
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"✓ Baseline embeddings device: {model.device}")

    # Test encoding
    test_text = "This is a test sentence"
    embedding = model.encode(test_text, convert_to_tensor=True)
    print(f"✓ Encoding successful, embedding device: {embedding.device}")
except Exception as e:
    print(f"⚠ Baseline embeddings test failed: {e}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

if torch.cuda.is_available():
    print("\n✅ GPU ACCELERATION IS ENABLED!")
    print(f"\nYour {torch.cuda.get_device_name(0)} is ready for:")
    print("  • Maniscope v2o: 6-290× speedup expected")
    print("  • HNSW v2o: 5-17× speedup expected")
    print("  • Jina Reranker v2o: 3-5× speedup expected")
    print("  • BGE-M3 v2o: 2-3× speedup expected")
    print("\nNext steps:")
    print("  1. Run: nvidia-smi")
    print("  2. Start Streamlit app")
    print("  3. Test Maniscope v2o in Eval ReRanker")
    print("  4. Monitor GPU usage and latency")
else:
    print("\n❌ GPU ACCELERATION NOT AVAILABLE")
    print("\nPlease check:")
    print("  1. PyTorch installation: pip show torch")
    print("  2. CUDA availability: nvidia-smi")
    print("  3. Re-install PyTorch with CUDA 12.1:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("=" * 70)
