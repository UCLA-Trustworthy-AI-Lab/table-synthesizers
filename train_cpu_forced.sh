#!/bin/bash
#
# Force CPU-only training with full multi-core utilization
# Use this when GPU is not supported (e.g., Blackwell sm_121)
#
# Usage:
#   ./train_cpu_forced.sh [training arguments]
#
# Example:
#   ./train_cpu_forced.sh --dataset insurance --models CTGAN --verbose
#   ./train_cpu_forced.sh --models CTGAN --cpu_cores 16 --dataset ./dataset/input_data --sample 1000 --use_wandb --file_type csv --iterate_datasets --verbose
#

echo "============================================================"
echo "CPU-Only Training with Full Multi-Core Utilization"
echo "============================================================"
echo

# Get CPU cores from argument or use system default
CPU_CORES=${CPU_CORES:-$(nproc)}

# Parse --cpu_cores argument if present
for arg in "$@"; do
    if [[ "$arg" =~ --cpu_cores ]]; then
        CPU_CORES=$(echo "$@" | grep -oP '(?<=--cpu_cores )\d+')
        break
    fi
done

# CRITICAL: Preload PyTorch's libgomp to avoid TLS conflicts
# This prevents "Assertion `listp != NULL' failed" error
TORCH_LIBGOMP=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libgomp.so.1'))" 2>/dev/null)
if [ -f "$TORCH_LIBGOMP" ]; then
    export LD_PRELOAD="$TORCH_LIBGOMP"
    echo "TLS Fix: Preloading $TORCH_LIBGOMP"
else
    echo "Warning: Could not find PyTorch's libgomp.so.1 for TLS fix"
fi

# CRITICAL: Disable CUDA to force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Set threading environment variables for maximum CPU utilization
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES
export NUMEXPR_NUM_THREADS=$CPU_CORES
export OPENBLAS_NUM_THREADS=$CPU_CORES
export TABSYN_NUM_THREADS=$CPU_CORES

# PyTorch threading
export PYTORCH_NUM_THREADS=$CPU_CORES

# DataLoader workers (use cores-1 to leave headroom)
DATALOADER_WORKERS=$((CPU_CORES - 1))
if [ $DATALOADER_WORKERS -lt 1 ]; then
    DATALOADER_WORKERS=1
fi
export DATALOADER_NUM_WORKERS=$DATALOADER_WORKERS

# Prevent any GPU memory allocation attempts
export PYTORCH_CUDA_ALLOC_CONF=""

echo "Environment configured for CPU training:"
echo "  CPU Cores: $CPU_CORES"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  DATALOADER_NUM_WORKERS: $DATALOADER_NUM_WORKERS"
echo "  CUDA_VISIBLE_DEVICES: (empty - GPU disabled)"
echo "  LD_PRELOAD: ${LD_PRELOAD:-none}"
echo

# Verify no GPU will be used
echo "Verifying PyTorch device:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device: CPU (forced)')"
echo

# Run the training script with all arguments passed through
echo "Starting training..."
echo
python train_all_compatible_models.py "$@"
