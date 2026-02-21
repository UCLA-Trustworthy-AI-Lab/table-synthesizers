# TLS Error Fix Guide

## Problem

**Error Message:**
```
Inconsistency detected by ld.so: dl-tls.c: 613: _dl_allocate_tls_init: Assertion `listp != NULL' failed!
```

**Root Cause:**
Multiple conflicting OpenMP libraries (libgomp) from different packages:
- PyTorch's libgomp.so.1
- Conda's libgomp.so.1
- xgboost's bundled libgomp
- scikit-learn's bundled libgomp

This causes TLS (Thread-Local Storage) initialization conflicts when training with GPU.

---

## Solutions

### Solution 1: Use Wrapper Script (RECOMMENDED)

Use the provided wrapper script that handles all environment variables:

```bash
# Make script executable (first time only)
chmod +x train_with_gpu_fix.sh

# Use it just like train_all_compatible_models.py
./train_with_gpu_fix.sh --dataset insurance --models CTGAN --verbose

# All arguments are passed through
./train_with_gpu_fix.sh \
    --data_folder /path/to/data \
    --iterate_datasets \
    --models CTGAN TVAE \
    --use_wandb
```

**What it does:**
- Preloads PyTorch's libgomp library
- Sets threading environment variables
- Reduces DataLoader workers to avoid conflicts

---

### Solution 2: Manual Environment Variables

Set environment variables before running Python:

```bash
# Set environment variables
export LD_PRELOAD=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libgomp.so.1'))")
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export DATALOADER_NUM_WORKERS=0

# Run training
python train_all_compatible_models.py --dataset insurance --models CTGAN
```

---

### Solution 3: Inline with Command

One-liner for quick runs:

```bash
LD_PRELOAD=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libgomp.so.1'))") \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python train_all_compatible_models.py --dataset insurance --models CTGAN
```

---

### Solution 4: Permanent Fix (Add to .bashrc or .zshrc)

Add to your shell profile for permanent effect:

```bash
# Add to ~/.bashrc or ~/.zshrc
export LD_PRELOAD=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libgomp.so.1'))" 2>/dev/null)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

---

## Testing the Fix

Test with verbose mode to confirm GPU is working:

```bash
./train_with_gpu_fix.sh \
    --dataset insurance \
    --models CTGAN \
    --epochs 1 \
    --samples 100 \
    --verbose \
    --output_dir ./test_tls_fix
```

**Expected output:**
```
[VERBOSE] PyTorch CUDA available: True
[VERBOSE] GPU detected: NVIDIA GB10
[VERBOSE] Model._device: cuda
[VERBOSE] Generator device: cuda:0
[VERBOSE] Training completed in X.Xs
```

**No TLS error should appear.**

---

## Performance Impact

These fixes have **minimal performance impact**:

- **LD_PRELOAD**: No performance impact, just ensures correct library is loaded
- **OMP_NUM_THREADS=1**: May reduce CPU parallelism slightly, but GPU is doing the heavy lifting anyway
- **DATALOADER_NUM_WORKERS=0**: Single-threaded data loading (still fast with GPU acceleration)

**Typical performance:**
- Before: 7 hours on CPU
- After (with fix): ~10 seconds on GPU ✅

---

## Troubleshooting

### If error persists:

1. **Check PyTorch installation:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

2. **Verify libgomp path:**
   ```bash
   python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libgomp.so.1'))"
   ls -la $(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libgomp.so.1'))")
   ```

3. **Check for other conflicts:**
   ```bash
   ldd $(python -c "import torch; print(torch.__file__.replace('__init__.py', 'lib/libtorch_cpu.so'))") | grep gomp
   ```

4. **Try reducing batch size** (less memory pressure):
   ```bash
   ./train_with_gpu_fix.sh --dataset insurance --models CTGAN --batch_size 100
   ```

---

## Alternative: Downgrade to Stable PyTorch

If the issue is too problematic, you can downgrade to stable PyTorch:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Note:** Stable PyTorch may not support Blackwell GPU (sm_121), so you'll need to test if it works with your hardware.

---

## Why This Happens

PyTorch nightly builds sometimes have more aggressive threading/parallelism that exposes conflicts between different OpenMP library versions. The LD_PRELOAD workaround ensures PyTorch's libgomp is loaded first, preventing conflicts with other packages' bundled versions.

---

## Summary

**Quick Fix:**
```bash
./train_with_gpu_fix.sh --dataset insurance --models CTGAN
```

**Result:** GPU training works without TLS errors! 🚀
