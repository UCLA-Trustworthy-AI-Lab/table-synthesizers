#!/bin/bash
#
# Script to check PyTorch CUDA installation and provide fix instructions
#

echo "======================================================================"
echo "PyTorch CUDA Installation Checker"
echo "======================================================================"
echo

# Check current PyTorch installation
echo "1. Current PyTorch Installation:"
python -c "import torch; print(f'   Version: {torch.__version__}'); print(f'   CUDA Available: {torch.cuda.is_available()}'); print(f'   CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>&1

# Check NVIDIA driver
echo
echo "2. NVIDIA Driver:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1 | awk -F',' '{print "   GPU: "$1; print "   Driver:"$2; print "   Memory:"$3}'
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "   CUDA Version from driver: $CUDA_VERSION"
else
    echo "   ERROR: nvidia-smi not found - NVIDIA driver may not be installed"
fi

# Check pip packages
echo
echo "3. PyTorch Related Packages:"
pip list 2>/dev/null | grep -E "torch|cuda" | sed 's/^/   /'

echo
echo "======================================================================"
echo "DIAGNOSIS"
echo "======================================================================"

PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>&1)
if [[ "$PYTORCH_VERSION" == *"+cpu"* ]]; then
    echo
    echo "❌ PROBLEM FOUND: PyTorch is CPU-only version!"
    echo
    echo "Your PyTorch installation does NOT have CUDA support."
    echo "Even though you have an NVIDIA GPU, PyTorch cannot use it."
    echo
    echo "======================================================================"
    echo "FIX INSTRUCTIONS"
    echo "======================================================================"
    echo
    echo "Run these commands to install PyTorch with CUDA 12.4 support:"
    echo
    echo "   # Uninstall CPU-only PyTorch"
    echo "   pip uninstall -y torch torchvision torchaudio"
    echo
    echo "   # Install CUDA-enabled PyTorch (CUDA 12.4 - most compatible)"
    echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    echo
    echo "Alternative: For CUDA 12.1 (if you have issues with 12.4):"
    echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    echo
    echo "After installation, verify with:"
    echo "   python -c \"import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\""
    echo
elif python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo
    echo "✅ PyTorch has CUDA support enabled!"
    python -c "import torch; print(f'   GPU: {torch.cuda.get_device_name(0)}'); print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
    echo
    echo "Your PyTorch installation looks correct."
else
    echo
    echo "⚠️  PyTorch installed but CUDA not detected."
    echo "   This might be due to:"
    echo "   - Driver/CUDA version mismatch"
    echo "   - PyTorch needs reinstalling"
    echo "   - Environment issues"
fi

echo
echo "======================================================================"
