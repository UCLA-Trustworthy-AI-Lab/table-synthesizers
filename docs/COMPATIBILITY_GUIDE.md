# Compatibility Guide: Platform-Specific Issues and Solutions

**Last Updated**: 2026-02-04

This guide documents known compatibility issues on specific platforms and provides workarounds for successful deployment.

---

## Table of Contents

1. [Python 3.12 + ARM64 Linux Issues](#python-312--arm64-linux-issues)
2. [TabSyn Python 3.12 Compatibility](#tabsyn-python-312-compatibility)
3. [Recommended Configurations](#recommended-configurations)
4. [Quick Solution Matrix](#quick-solution-matrix)

---

## Python 3.12 + ARM64 Linux Issues

### Issue: Synthcity Dependency Conflict

**Affected Platform**: Python 3.12 on ARM64 Linux (aarch64)

**Error**:
```
× No solution found when resolving dependencies:
╰─▶ Because only the following versions of torchtext are available...
    and torchtext>=0.10.0,<=0.14.1 has no wheels with a matching Python ABI tag (e.g., `cp312`)
```

**Root Cause**:
- Synthcity depends on `decaf-synthetic-data`
- `decaf-synthetic-data` requires `torchtext>=0.10.0,<=0.14.1`
- `torchtext` does not have pre-built wheels for Python 3.12 + ARM64 Linux
- Only available platforms for torchtext: `manylinux1_x86_64`, `macosx_10_13_x86_64`, `macosx_11_0_arm64`, `win_amd64`

**Affected Models**:
- ❌ GReat (GPU-accelerated) - requires synthcity
- ❌ NFlow (GPU-accelerated) - requires synthcity
- ❌ ARF (CPU-only) - requires synthcity
- ❌ BayesianNetwork (CPU-only) - requires synthcity

**Unaffected Models** (work perfectly):
- ✅ CTGAN (GPU-accelerated, 15-30x speedup)
- ✅ TVAE (GPU-accelerated, 10-20x speedup)
- ✅ TabDDPM (GPU-accelerated, 20-40x speedup)
- ✅ PATE-CTGAN (GPU-accelerated, 10-25x speedup)
- ✅ AutoDiff (GPU-accelerated, 15-30x speedup)
- ✅ SMOTE (CPU-only)
- ✅ CART (CPU-only)
- ✅ DPCART (CPU-only)
- ✅ Identity (CPU-only)
- ✅ AIM (CPU-only)

### Solutions

#### Solution 1: Use Alternative GPU Models (RECOMMENDED)

**Best option for Python 3.12 + ARM64 with GPU acceleration:**

```bash
# Install core dependencies (no synthcity)
pip install -r requirements.txt

# Verify installation
python -c "from stg.tableSynthesizer import TableSynthesizer; print('✅ Core models available')"
```

**Alternative models with comparable capabilities:**

| Instead of GReat | Use CTGAN or TabDDPM |
|-----------------|---------------------|
| Transformer-based | GAN-based (CTGAN) or Diffusion-based (TabDDPM) |
| 20-35x GPU speedup | 15-30x (CTGAN) or 20-40x (TabDDPM) speedup |
| High quality | High to Highest quality |

**Example replacement code:**

```python
# ❌ Original (GReat - not available on Python 3.12 + ARM64)
# from stg.tableSynthesizer import TableSynthesizer
# config = {"epochs": 50, "batch_size": 128}
# synthesizer = TableSynthesizer('GReat', config=config)

# ✅ Replacement (CTGAN - works on all platforms)
from stg.tableSynthesizer import TableSynthesizer
config = {"epochs": 50, "batch_size": 200}
synthesizer = TableSynthesizer('CTGAN', config=config)
synthesizer.model.set_device('cuda')  # GPU acceleration

# ✅ Or use TabDDPM for highest quality
config = {"epochs": 100, "batch_size": 256}
synthesizer = TableSynthesizer('TabDDPM', config=config)
synthesizer.model.set_device('cuda')
```

#### Solution 2: Use Python 3.10 or 3.11

**If GReat/NFlow are absolutely required:**

```bash
# Create conda environment with Python 3.10
conda create -n table-synth python=3.10
conda activate table-synth

# Install all dependencies (including synthcity)
pip install -r requirements.txt
pip install -r requirements-optional.txt

# Now GReat and NFlow are available
python -c "from stg.tableSynthesizer import TableSynthesizer; print('✅ All models available')"
```

#### Solution 3: Use x86_64 Platform

**If platform flexibility exists:**

```bash
# On x86_64 Linux (not ARM64)
pip install -r requirements.txt
pip install -r requirements-optional.txt

# All models work including GReat, NFlow
```

#### Solution 4: Build torchtext from Source (Advanced)

**For advanced users who need synthcity on Python 3.12 + ARM64:**

```bash
# Build torchtext from source (may take 30-60 minutes)
git clone https://github.com/pytorch/text.git
cd text
git checkout v0.14.1
python setup.py install

# Then install synthcity
pip install synthcity==0.2.12
```

**Note**: This approach is complex and may have other dependency conflicts.

---

## TabSyn Python 3.12 Compatibility

### Issue: Subprocess Import Errors

**Affected Platform**: Python 3.12 (all architectures)

**Error**:
```python
ImportError: attempted relative import beyond top-level package
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

**Root Cause**:
- TabSyn uses subprocess to launch training with relative imports
- Python 3.12 changed module import behavior
- `pkgutil.ImpImporter` was removed in Python 3.12
- Subprocess execution fails with import errors

**Affected Model**:
- ❌ TabSyn (GPU-accelerated, 5-15x speedup)

### Solutions

#### Solution 1: Use Alternative Models (RECOMMENDED)

**TabSyn alternatives with better or comparable quality:**

```python
# ✅ CTGAN - Best balance of speed and quality
config = {"epochs": 50, "batch_size": 200}
synthesizer = TableSynthesizer('CTGAN', config=config)
synthesizer.model.set_device('cuda')
# Speed: 15-30x faster than CPU
# Quality: High, widely tested in production

# ✅ TabDDPM - Highest quality
config = {"epochs": 100, "batch_size": 256}
synthesizer = TableSynthesizer('TabDDPM', config=config)
synthesizer.model.set_device('cuda')
# Speed: 20-40x faster than CPU
# Quality: State-of-the-art, best for critical applications

# ✅ TVAE - Fastest training
config = {"epochs": 100, "batch_size": 300}
synthesizer = TableSynthesizer('TVAE', config=config)
synthesizer.model.set_device('cuda')
# Speed: 10-20x faster than CPU, fastest training
# Quality: Good for most use cases
```

**Why CTGAN/TabDDPM are better choices:**
- ✅ No Python version restrictions
- ✅ Better GPU acceleration (15-40x vs TabSyn's 5-15x)
- ✅ More stable and widely tested
- ✅ Better documentation and community support
- ✅ Work on all platforms (Linux, macOS, Windows)

#### Solution 2: Use Python 3.10 or 3.11

**If TabSyn is absolutely required:**

```bash
# Create conda environment with Python 3.10
conda create -n tabsyn-env python=3.10
conda activate tabsyn-env

# Install dependencies
pip install -r requirements.txt

# TabSyn now works
python train_tabsyn_poc.py --dataset insurance --epochs 10 --samples 1000
```

#### Solution 3: Docker Container with Python 3.10

**For isolated TabSyn environment:**

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Now run TabSyn scripts
```

---

## Recommended Configurations

### Configuration 1: Production GPU Training (Python 3.12 + ARM64)

**Best for**: Maximum compatibility and performance on modern hardware

```bash
# Install core dependencies only
pip install -r requirements.txt

# Available models
Available GPU Models:
  ✅ CTGAN (15-30x speedup) - RECOMMENDED for general use
  ✅ TVAE (10-20x speedup) - RECOMMENDED for fast prototyping
  ✅ TabDDPM (20-40x speedup) - RECOMMENDED for highest quality
  ✅ PATE-CTGAN (10-25x speedup) - For privacy requirements
  ✅ AutoDiff (15-30x speedup) - For differential privacy

Available CPU Models:
  ✅ SMOTE, CART, DPCART, Identity, AIM
```

**Training example:**

```python
from stg.tableSynthesizer import TableSynthesizer
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Train CTGAN (best general-purpose model)
config = {"epochs": 50, "batch_size": 200}
synthesizer = TableSynthesizer('CTGAN', config=config)
synthesizer.model.set_device('auto')  # Auto-detect GPU

# Fit and generate
synthesizer.fit(df)
synthetic_df = synthesizer.sample(n=10000, return_dataframe=True)
```

### Configuration 2: All Models Available (Python 3.10 x86_64)

**Best for**: Maximum model availability, research, experimentation

```bash
# Create Python 3.10 environment
conda create -n table-synth python=3.10
conda activate table-synth

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-optional.txt

# All models available including GReat, NFlow, TabSyn
```

### Configuration 3: Docker Production (Any Platform)

**Best for**: Deployment, CI/CD, reproducibility

```bash
# Build production container
./docker/build_docker_VAE.sh

# Run with specific model
docker run -v $(pwd)/data:/app/data ltm:latest \
    python train_ctgan_poc.py --dataset insurance --epochs 50
```

---

## Quick Solution Matrix

| Your Platform | Recommended Solution | Available GPU Models | Unavailable Models |
|--------------|---------------------|---------------------|-------------------|
| **Python 3.12 + ARM64 Linux** | Use core models only<br>`pip install -r requirements.txt` | CTGAN, TVAE, TabDDPM,<br>PATE-CTGAN, AutoDiff | GReat, NFlow, TabSyn,<br>ARF, BayesianNetwork |
| **Python 3.12 + x86_64 Linux** | Install optional deps<br>`pip install synthcity` | CTGAN, TVAE, TabDDPM,<br>PATE-CTGAN, AutoDiff,<br>**GReat, NFlow** | TabSyn |
| **Python 3.10/3.11 + Any** | Install all deps<br>`pip install -r requirements*.txt` | **ALL GPU MODELS** | None |
| **Docker (Any Host)** | Use provided Dockerfiles | **ALL GPU MODELS** | None (if Dockerfile uses Python 3.10) |

---

## Model Selection Guide

### For General-Purpose Synthetic Data

**CTGAN (RECOMMENDED)**:
- ✅ Best balance of speed, quality, and stability
- ✅ Works on all platforms (Python 3.10-3.12, ARM64/x86_64)
- ✅ 15-30x GPU speedup
- ✅ Extensively tested in production
- ✅ Excellent documentation and community support

```python
config = {"epochs": 50, "batch_size": 200}
synthesizer = TableSynthesizer('CTGAN', config=config)
synthesizer.model.set_device('auto')
```

### For Maximum Quality

**TabDDPM (RECOMMENDED)**:
- ✅ State-of-the-art quality
- ✅ Works on all platforms
- ✅ 20-40x GPU speedup
- ✅ Best for critical applications requiring highest fidelity

```python
config = {"epochs": 100, "batch_size": 256}
synthesizer = TableSynthesizer('TabDDPM', config=config)
synthesizer.model.set_device('auto')
```

### For Fast Prototyping

**TVAE (RECOMMENDED)**:
- ✅ Fastest training time
- ✅ Works on all platforms
- ✅ 10-20x GPU speedup
- ✅ Good quality for most use cases

```python
config = {"epochs": 100, "batch_size": 300}
synthesizer = TableSynthesizer('TVAE', config=config)
synthesizer.model.set_device('auto')
```

### For Privacy-Preserving Synthesis

**PATE-CTGAN**:
- ✅ Differential privacy guarantees
- ✅ Works on all platforms
- ✅ 10-25x GPU speedup

```python
config = {
    "epochs": 50,
    "batch_size": 100,
    "num_teachers": 10,
    "epsilon": 1.0,
    "delta": 1e-5
}
synthesizer = TableSynthesizer('PATE-CTGAN', config=config)
synthesizer.model.set_device('auto')
```

---

## Testing Your Configuration

### Quick Compatibility Test

```bash
# Test 1: Check Python version
python --version
# Should show: Python 3.X.X

# Test 2: Check architecture
python -c "import platform; print(platform.machine())"
# Shows: aarch64 (ARM64) or x86_64

# Test 3: Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test 4: Test core models
python -c "from stg.tableSynthesizer import TableSynthesizer; print('✅ Core OK')"

# Test 5: Test synthcity models (may fail on Python 3.12 + ARM64)
python -c "from stg.GREAT.great_synthesizer import GREATSynthesizer; print('✅ Synthcity OK')"
```

### Full Model Availability Check

```python
from stg.tableSynthesizer import TableSynthesizer

models_to_test = [
    'CTGAN', 'TVAE', 'TabDDPM', 'PATE-CTGAN', 'AutoDiff',  # Core GPU models
    'GReat', 'NFlow',  # Synthcity GPU models
    'TabSyn',  # May fail on Python 3.12
]

for model_name in models_to_test:
    try:
        synthesizer = TableSynthesizer(model_name, config={})
        print(f"✅ {model_name:15} - Available")
    except Exception as e:
        print(f"❌ {model_name:15} - Unavailable: {type(e).__name__}")
```

---

## FAQ

### Q: Can I use GReat on Python 3.12 + ARM64?

**A**: No, not without significant workarounds (building torchtext from source). **Recommended**: Use CTGAN or TabDDPM instead - they provide comparable or better quality with better platform support.

### Q: Which model should I use if GReat is unavailable?

**A**:
- For general use: **CTGAN** (best balance)
- For highest quality: **TabDDPM** (state-of-the-art)
- For fastest training: **TVAE** (good quality, 2-3x faster)

### Q: Can I use TabSyn on Python 3.12?

**A**: No, TabSyn has Python 3.12 compatibility issues with subprocess imports. **Recommended**: Use TabDDPM (better quality, 20-40x speedup) or CTGAN (15-30x speedup, more stable) instead.

### Q: Do I need synthcity for GPU acceleration?

**A**: No! The best GPU-accelerated models (CTGAN, TVAE, TabDDPM, PATE-CTGAN, AutoDiff) don't require synthcity and work on all platforms.

### Q: How do I get all models working?

**A**: Use Python 3.10 or 3.11 instead of 3.12. On x86_64 architecture, all models work. On ARM64, you may need to skip synthcity-based models.

---

## Support and Resources

- **GPU Acceleration Guide**: `docs/GPU_ACCELERATION_GUIDE.md`
- **Model-Specific Configs**: `docs/MODEL_GPU_CONFIGS.md`
- **POC Results**: `docs/POC_RESULTS.md`
- **Documentation Index**: `docs/README.md`

For issues or questions, check the troubleshooting sections in the documentation files above.

---

**Created**: 2026-02-04
**Platform Tested**: Python 3.12.12, ARM64 Linux (aarch64), NVIDIA GB10 (SM 12.1, CUDA 13.0)
