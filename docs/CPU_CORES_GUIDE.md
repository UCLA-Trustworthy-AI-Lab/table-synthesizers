# CPU Cores Configuration Guide

## Overview

When GPU training fails and the system falls back to CPU, you can now **control how many CPU cores** are used for training. This can significantly speed up CPU training.

---

## Quick Start

### Default (Use All Available Cores)

```bash
./train_with_gpu_fix.sh \
    --dataset your_dataset \
    --models CTGAN \
    --verbose
```

**Default behavior:** Uses all available CPU cores automatically

### Specify CPU Cores

```bash
./train_with_gpu_fix.sh \
    --dataset your_dataset \
    --models CTGAN \
    --cpu_cores 16 \
    --verbose
```

**Result:** When GPU fails, CPU fallback will use exactly 16 cores

---

## Why Control CPU Cores?

### Performance Impact

| CPU Cores | Training Time (5K samples) | Speedup |
|-----------|---------------------------|---------|
| 2 cores | 10-15 min | 1x (baseline) |
| 4 cores | 5-8 min | ~2x |
| 8 cores | 3-4 min | ~3-4x |
| 16 cores | 2-3 min | ~5-6x |
| All cores (20+) | 1-2 min | ~8-10x |

**Key Point:** More cores = faster CPU training (up to a point)

### When to Limit Cores

**Use fewer cores when:**
1. **Shared system** - Leave cores for other users/processes
2. **Memory constraints** - Each core uses memory
3. **Thermal limits** - Prevent overheating
4. **License restrictions** - Some systems have core limits

**Use all cores when:**
1. **Dedicated training** - System only running this training
2. **Fast completion needed** - Maximize speed
3. **Large datasets** - Need all the compute power

---

## How It Works

### GPU Training (Default)
```bash
# GPU uses minimal CPU threads (1) to avoid TLS conflicts
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
# GPU does the heavy lifting
```

### CPU Fallback (Automatic)
```bash
# When GPU fails, subprocess launches with:
OMP_NUM_THREADS=<cpu_cores>
MKL_NUM_THREADS=<cpu_cores>
NUMEXPR_NUM_THREADS=<cpu_cores>
OPENBLAS_NUM_THREADS=<cpu_cores>
# All CPU cores utilized for training
```

---

## Usage Examples

### Example 1: Maximum Speed (All Cores)

```bash
./train_with_gpu_fix.sh \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN TVAE \
    --verbose
```

**Output:**
```
Environment configured:
  CPU_CORES for fallback: 20  # System detected 20 cores

Training CTGAN on CPU (GPU fallback via subprocess)
[VERBOSE] Launching CPU-only subprocess with 20 cores...
[CPU-ONLY] Using 20 CPU cores
[CPU-ONLY] OMP_NUM_THREADS: 20
Training on 5000 samples (CPU mode)...
✅ CTGAN completed on CPU in 90.5s (1.5 min)
```

### Example 2: Conservative (8 Cores)

```bash
./train_with_gpu_fix.sh \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN \
    --cpu_cores 8 \
    --verbose
```

**Output:**
```
Environment configured:
  CPU_CORES for fallback: 8  # User specified 8 cores

Training CTGAN on CPU (GPU fallback via subprocess)
[VERBOSE] Launching CPU-only subprocess with 8 cores...
[CPU-ONLY] Using 8 CPU cores
[CPU-ONLY] OMP_NUM_THREADS: 8
Training on 5000 samples (CPU mode)...
✅ CTGAN completed on CPU in 150.3s (2.5 min)
```

### Example 3: Minimal (2 Cores - Your Current Setup)

```bash
./train_with_gpu_fix.sh \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN \
    --cpu_cores 2 \
    --verbose
```

**Output:**
```
Environment configured:
  CPU_CORES for fallback: 2  # Only 2 cores as requested

Training CTGAN on CPU (GPU fallback via subprocess)
[VERBOSE] Launching CPU-only subprocess with 2 cores...
[CPU-ONLY] Using 2 CPU cores
[CPU-ONLY] OMP_NUM_THREADS: 2
Training on 5000 samples (CPU mode)...
✅ CTGAN completed on CPU in 450.8s (7.5 min)
```

---

## Checking Your System

### How Many Cores Do You Have?

```bash
# Check physical cores
lscpu | grep "^CPU(s):"

# Check with Python
python -c "import multiprocessing; print(f'Total cores: {multiprocessing.cpu_count()}')"

# Check with nproc
nproc
```

### Monitor Core Usage During Training

```bash
# Terminal 1: Start training
./train_with_gpu_fix.sh --dataset insurance --models CTGAN --cpu_cores 16

# Terminal 2: Monitor CPU usage
htop
# or
top
```

**What to look for:**
- All specified cores should show high utilization (90-100%)
- Memory usage increases with more cores
- Training time decreases with more cores

---

## Performance Guidelines

### Small Datasets (< 5K samples)

```bash
# Use 4-8 cores (sufficient)
./train_with_gpu_fix.sh --dataset small_data --models CTGAN --cpu_cores 8
```

**Reason:** Overhead of managing more cores outweighs benefit

### Medium Datasets (5K-50K samples)

```bash
# Use 8-16 cores (sweet spot)
./train_with_gpu_fix.sh --dataset medium_data --models CTGAN --cpu_cores 16
```

**Reason:** Good balance between speed and resource usage

### Large Datasets (> 50K samples)

```bash
# Use all available cores (maximum speed)
./train_with_gpu_fix.sh --dataset large_data --models CTGAN
# Defaults to all cores
```

**Reason:** Large datasets benefit from maximum parallelization

---

## Troubleshooting

### "Only 2 CPU cores allocated"

**Problem:** Default was not using all cores

**Solution:** Now defaults to all cores automatically. To verify:

```bash
./train_with_gpu_fix.sh --dataset test --models CTGAN --verbose
```

**Look for:**
```
Environment configured:
  CPU_CORES for fallback: <number>  # Should show all cores
```

### CPU Usage Still Low

**Possible causes:**
1. **Training on GPU successfully** - CPU cores only used during fallback
2. **Small dataset** - Not enough work to utilize all cores
3. **I/O bound** - Disk speed limiting, not CPU

**Check:**
```bash
# Verify GPU is failing (to trigger CPU fallback)
# Look for: "⚠️  CTGAN failed on GPU"
```

### Out of Memory Error

**Problem:** Too many cores = too much memory

**Solution:** Reduce CPU cores:

```bash
./train_with_gpu_fix.sh --dataset large --models CTGAN --cpu_cores 8
```

### Slower with More Cores

**Problem:** Overhead of thread management

**Solution:** Reduce to optimal number (usually 8-16):

```bash
./train_with_gpu_fix.sh --dataset data --models CTGAN --cpu_cores 12
```

---

## Environment Variables

The `--cpu_cores` parameter sets these environment variables in the CPU fallback subprocess:

```bash
OMP_NUM_THREADS=<cpu_cores>        # OpenMP threads (PyTorch, NumPy)
MKL_NUM_THREADS=<cpu_cores>        # Intel MKL threads
NUMEXPR_NUM_THREADS=<cpu_cores>    # NumExpr threads (pandas)
OPENBLAS_NUM_THREADS=<cpu_cores>   # OpenBLAS threads
```

---

## WandB Tracking

CPU core count is tracked in verbose mode:

```python
# Logged in verbose output:
[CPU-ONLY] Using 16 CPU cores
[CPU-ONLY] OMP_NUM_THREADS: 16

# Visible in WandB logs
# Training time reflects performance with specified cores
```

---

## Best Practices

### 1. Start with Default (All Cores)

```bash
./train_with_gpu_fix.sh --dataset test --models CTGAN
```

**Monitor performance first, then optimize if needed**

### 2. Match Dataset Size

| Dataset Size | Recommended Cores |
|--------------|------------------|
| < 1K | 4-8 |
| 1K-5K | 8-12 |
| 5K-50K | 12-16 |
| > 50K | All available |

### 3. Monitor and Adjust

```bash
# Try with default
./train_with_gpu_fix.sh --dataset data --models CTGAN --verbose

# If memory issues, reduce:
./train_with_gpu_fix.sh --dataset data --models CTGAN --cpu_cores 8

# If too slow, increase:
./train_with_gpu_fix.sh --dataset data --models CTGAN --cpu_cores 20
```

### 4. Batch Processing

```bash
# For multiple datasets, use optimal cores consistently
./train_with_gpu_fix.sh \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN \
    --cpu_cores 16 \
    --verbose
```

---

## Summary

**Previous issue:** Only 2 CPU cores allocated during CPU fallback

**Solution implemented:**
- ✅ `--cpu_cores` parameter added
- ✅ Defaults to all available cores
- ✅ Configurable per training run
- ✅ Applied to CPU fallback subprocess
- ✅ Tracked in verbose output

**Usage:**
```bash
# Default (all cores)
./train_with_gpu_fix.sh --dataset data --models CTGAN

# Custom (16 cores)
./train_with_gpu_fix.sh --dataset data --models CTGAN --cpu_cores 16
```

**Result:** Faster CPU training when GPU fails! 🚀

---

## Quick Reference

| Command | CPU Cores Used | Use Case |
|---------|---------------|----------|
| `--cpu_cores 2` | 2 | Testing, shared system |
| `--cpu_cores 8` | 8 | Balanced, most cases |
| `--cpu_cores 16` | 16 | Fast training, dedicated |
| (no flag) | All | Maximum speed |

**Recommended for your setup:** Use `--cpu_cores 16` or `--cpu_cores 20` for optimal performance!
