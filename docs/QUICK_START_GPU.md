# Quick Start - GPU Training

## TL;DR

Your Blackwell GB10 GPU (sm_121) has **limited support** in current PyTorch. The training script **automatically falls back to CPU** when GPU fails. Everything works, some operations are just slower.

---

## How to Train (3 Steps)

### 1. Use the Wrapper Script

```bash
./train_with_gpu_fix.sh [your arguments]
```

This replaces `python train_all_compatible_models.py`.

### 2. Add Verbose Flag (Optional but Recommended)

```bash
./train_with_gpu_fix.sh --verbose [other args]
```

Lets you see if GPU is being used or if it falls back to CPU.

### 3. Run Your Training

```bash
# Single dataset
./train_with_gpu_fix.sh \
    --dataset insurance \
    --models CTGAN TVAE \
    --verbose

# Multiple datasets
./train_with_gpu_fix.sh \
    --data_folder /path/to/datasets \
    --iterate_datasets \
    --models CTGAN TVAE \
    --use_wandb \
    --verbose
```

---

## What You'll See

### If GPU Works (Fast)
```
[VERBOSE] Model._device: cuda
Training on 1070 samples...
[VERBOSE] Training completed in 10.2s
✅ CTGAN completed in 10.2s
```

### If GPU Fails → CPU Fallback (Slower but Works)
```
Training on 3871 samples...
⚠️  CTGAN failed on GPU: RuntimeError: no kernel image is available
🔄 Retrying with CPU fallback...

Training CTGAN on CPU (GPU fallback)
Training on 3871 samples (CPU mode)...
✅ CTGAN completed on CPU in 45.3s
```

---

## Performance Guide

| Dataset Size | GPU (when works) | CPU (fallback) |
|--------------|------------------|----------------|
| < 5K samples | 5-10s | 30-60s |
| 5K-50K samples | 10-30s | 2-5 min |
| > 50K samples | 30-60s | 10-30 min |

---

## Common Scenarios

### All Datasets Train Successfully
✅ Perfect! GPU is working well with your data.

### Some Datasets Use GPU, Some Use CPU
✅ Normal! Certain operations trigger CPU fallback. Training still completes.

### All Datasets Fall Back to CPU
⚠️ GPU isn't compatible yet with PyTorch's current kernels. Training still works, just slower. Check for PyTorch updates monthly.

---

## Files You Need to Know

- **train_with_gpu_fix.sh** - Main wrapper script (use this instead of .py)
- **TLS_FIX_GUIDE.md** - Detailed info about OpenMP library fixes
- **GPU_COMPATIBILITY_GUIDE.md** - Complete guide to Blackwell GPU issues
- **This file** - Quick reference

---

## Troubleshooting

### "cannot execute: required file not found"
```bash
sed -i 's/\r$//' train_with_gpu_fix.sh
chmod +x train_with_gpu_fix.sh
```

### Still getting TLS errors
```bash
# Check environment is set
./train_with_gpu_fix.sh --verbose --dataset insurance --models CTGAN

# Look for:
# "LD_PRELOAD: /path/to/libgomp.so.1"
```

### Want to verify GPU is detected
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected:**
```
CUDA: True
GPU: NVIDIA GB10
```

---

## Summary

**Problem:** Blackwell GPU (sm_121) not fully supported by PyTorch yet

**Solution:** Automatic CPU fallback implemented

**Result:** Training always completes, sometimes fast (GPU), sometimes slower (CPU)

**Action:** Just use `./train_with_gpu_fix.sh` and let the script handle everything!

---

## Full Documentation

- **TLS_FIX_GUIDE.md**: OpenMP library conflict fixes
- **GPU_COMPATIBILITY_GUIDE.md**: Complete Blackwell GPU guide
- **train_all_compatible_models.py**: Now has automatic CPU fallback logic

---

## Examples

### Basic Training
```bash
./train_with_gpu_fix.sh --dataset insurance --models CTGAN --verbose
```

### Batch Process Multiple Datasets
```bash
./train_with_gpu_fix.sh \
    --data_folder ./datasets \
    --iterate_datasets \
    --file_type csv \
    --models CTGAN TVAE TabDDPM \
    --verbose
```

### With WandB Tracking
```bash
# First: set up .env with WANDB_API_KEY
./train_with_gpu_fix.sh \
    --dataset insurance \
    --models CTGAN \
    --use_wandb \
    --wandb_project my-project \
    --verbose
```

### Custom Config Directory
```bash
./train_with_gpu_fix.sh \
    --config_dir ./my_configs \
    --dataset insurance \
    --models CTGAN \
    --verbose
```

---

**Questions?** Check the detailed guides:
- TLS errors → TLS_FIX_GUIDE.md
- GPU issues → GPU_COMPATIBILITY_GUIDE.md
- Configuration → CONFIG_EXTERNALIZATION_SUMMARY.md
- Batch processing → ITERATIVE_TRAINING_GUIDE.md
- Metrics → WANDB_INTEGRATION_GUIDE.md
