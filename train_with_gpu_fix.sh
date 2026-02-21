#!/bin/bash
#
# Wrapper script to fix TLS/OpenMP library conflicts when training with GPU
#
# Usage:
#   ./train_with_gpu_fix.sh [training arguments]
#
# Example:
#   ./train_with_gpu_fix.sh --dataset insurance --models CTGAN --verbose
#

echo "============================================================"
echo "GPU Training with TLS Fix and Blackwell Compatibility"
echo "============================================================"
echo

# Parse CPU cores argument (default to system CPU count)
CPU_CORES=${CPU_CORES:-$(nproc)}

# Preload PyTorch's libgomp to avoid conflicts
export LD_PRELOAD=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libgomp.so.1'))")

# Set threading environment variables
# For GPU: limit to 1 to avoid TLS conflicts
# For CPU fallback: will be overridden to use more cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Force PyTorch to use sm_120 kernels for sm_121 GPU
export TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 11.0 12.0"
export CUDA_VISIBLE_DEVICES=0

# Enable blocking mode for better error messages
export CUDA_LAUNCH_BLOCKING=1

# Set DataLoader to use fewer workers (reduces thread conflicts)
export DATALOADER_NUM_WORKERS=0

# Try to force compute mode compatibility
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Environment configured:"
echo "  LD_PRELOAD: $LD_PRELOAD"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS (GPU mode)"
echo "  CPU_CORES for fallback: $CPU_CORES"
echo "  DATALOADER_NUM_WORKERS: $DATALOADER_NUM_WORKERS"
echo

# Run the training script with all arguments passed through
python train_all_compatible_models.py "$@"
