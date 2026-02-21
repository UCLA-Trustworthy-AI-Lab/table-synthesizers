# Documentation Index

Complete documentation for the table-synthesizers project. For a quick overview, see the root [README.md](../README.md).

## Getting Started

| Document | Description |
|----------|-------------|
| [Root README](../README.md) | Project overview, installation, and quick start |
| [CLAUDE.md (root)](../CLAUDE.md) | Development guide and project constraints |
| [Development Guide](CLAUDE.md) | Detailed architecture, API reference, and contributing guide |

## GPU & Hardware

| Document | Description |
|----------|-------------|
| [GPU Acceleration Guide](GPU_ACCELERATION_GUIDE.md) | Comprehensive GPU setup, supported architectures, optimization |
| [GPU Quick Start](QUICK_START_GPU.md) | Fast GPU setup for common hardware |
| [GPU Compatibility Guide](GPU_COMPATIBILITY_GUIDE.md) | GPU compatibility matrix across platforms |
| [Blackwell GPU Solution](BLACKWELL_GPU_FINAL_SOLUTION.md) | NVIDIA Blackwell (GB10/B100/B200) specific setup |
| [Model GPU Configs](MODEL_GPU_CONFIGS.md) | Per-model GPU configurations and memory requirements |
| [CPU Cores Guide](CPU_CORES_GUIDE.md) | CPU optimization and multi-core usage |

## Configuration & Training

| Document | Description |
|----------|-------------|
| [Config System](../config/README.md) | JSON configuration files and customization |
| [Config Externalization Summary](CONFIG_EXTERNALIZATION_SUMMARY.md) | How the config system was designed |
| [Enhanced Training Guide](ENHANCED_TRAINING_GUIDE.md) | Training best practices and advanced options |
| [Iterative Training Guide](ITERATIVE_TRAINING_GUIDE.md) | Multi-dataset iterative training workflows |
| [Iterative Training Summary](ITERATIVE_TRAINING_SUMMARY.md) | Summary of iterative training features |
| [Batch Size Optimization](BATCH_SIZE_OPTIMIZATION_SUMMARY.md) | GPU-aware batch size tuning |
| [Hyperparameter Analysis](HYPERPARAMETER_ANALYSIS.md) | Hyperparameter tuning guide |

## Monitoring & Integration

| Document | Description |
|----------|-------------|
| [WandB Integration Guide](WANDB_INTEGRATION_GUIDE.md) | Setting up Weights & Biases tracking |
| [WandB Integration Summary](WANDB_INTEGRATION_SUMMARY.md) | Summary of WandB features |

## Model Comparisons & References

| Document | Description |
|----------|-------------|
| [CTGAN vs PATECTGAN](CTGAN_vs_PATECTGAN_Comparison.md) | Comparison of GAN-based synthesizers |
| [Compatibility Guide](COMPATIBILITY_GUIDE.md) | Platform and Python version compatibility |
| [Platform Compatibility Fixes](PLATFORM_COMPATIBILITY_FIX_SUMMARY.md) | Fixes for cross-platform issues |

## Proof of Concept & Results

| Document | Description |
|----------|-------------|
| [POC Summary](POC_SUMMARY.md) | Proof of concept overview |
| [POC Results](POC_RESULTS.md) | Detailed POC testing results |
| [TabSyn POC Guide](TABSYN_POC_GUIDE.md) | TabSyn-specific proof of concept |
| [Test Validation Report](TEST_VALIDATION_REPORT.md) | Test validation results |

## Operational

| Document | Description |
|----------|-------------|
| [TLS Fix Guide](TLS_FIX_GUIDE.md) | TLS/SSL certificate troubleshooting |
| [Changes Summary](CHANGES_SUMMARY.md) | Changelog of recent updates |
| [Implementation Plan](implementation_plan.md) | Project roadmap and planning |

## Quick Reference

### Available Models (19 total)

**GPU-Accelerated (10-50x speedup)**:
CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, GREAT, NFlow, TabSyn, LTM_VAE, TabPFGen, TabDiff

**CPU-Only**:
Identity, CART, DPCART, SMOTE, AIM, BayesianNetwork, ARF, GaussianCopula

### Installation (4-Tier Requirements)

```bash
# Base only (Identity, CART, DPCART, SMOTE, AIM - no torch needed)
pip install -r requirements.txt

# GPU models on Blackwell (CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, TabSyn, TabPFGen)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-gpu.txt

# CPU torch models (same GPU models, running on CPU)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt

# Synthcity models (BayesianNetwork, ARF, GREAT, NFlow, TabDiff - best on Python 3.11)
pip install -r requirements-synthcity.txt

# SDV models (GaussianCopula)
pip install sdv>=1.34.0
```

| File | Provides | Key Packages |
|------|----------|-------------|
| `requirements.txt` | Base (CPU-only models) | pandas, numpy, sklearn, scipy |
| `requirements-gpu.txt` | GPU models (CUDA SM 12.1) | torch>=2.7, torchvision, transformers |
| `requirements-cpu.txt` | GPU models on CPU | torch>=2.7 (CPU wheel), transformers |
| `requirements-synthcity.txt` | Synthcity models | synthcity>=0.2.12 |
| `sdv>=1.34.0` | GaussianCopula | sdv |

### Model Name Aliases

The training script accepts case-insensitive model names and common aliases:

```bash
# All equivalent
python train_all_compatible_models.py --models CTGAN
python train_all_compatible_models.py --models ctgan
python train_all_compatible_models.py --models BayesianNetwork
python train_all_compatible_models.py --models BN           # shorthand
python train_all_compatible_models.py --models BaysianNetwork  # common typo
python train_all_compatible_models.py --models PATE-CTGAN   # config name
python train_all_compatible_models.py --models PATECTGAN    # registry name
python train_all_compatible_models.py --models GreaT        # paper name
python train_all_compatible_models.py --models great        # lowercase
```

### Synthcity Model Hyperparameters

Synthcity-based models (GREAT, BayesianNetwork, ARF, NFlow) accept plugin-specific
hyperparameters via the config dict. These are passed directly to synthcity's plugin
constructor.

```python
# GREAT - Transformer-based (distilgpt2 by default)
synth = TableSynthesizer('GREAT', config={
    'n_iter': 200,          # Training epochs (default: 100)
    'llm': 'distilgpt2',   # Language model
    'batch_size': 16,       # Training batch size (default: 8)
    'device': 'cuda',       # cpu or cuda
})

# BayesianNetwork - Structure learning with DAG
synth = TableSynthesizer('BayesianNetwork', config={
    'struct_learning_search_method': 'hillclimb',  # hillclimb, pc, tree_search
    'struct_learning_score': 'bic',                # bdeu, bds, bic, k2
    'struct_max_indegree': 6,                      # Max parent nodes (default: 4)
    'encoder_max_clusters': 20,                    # Discretization clusters (default: 10)
})

# ARF - Adversarial Random Forest
synth = TableSynthesizer('ARF', config={
    'num_trees': 50,        # Number of trees (default: 30)
    'max_iters': 20,        # Max iterations (default: 10)
    'min_node_size': 10,    # Min leaf size (default: 5)
})

# NFlow - Normalizing Flows
synth = TableSynthesizer('NFlow', config={
    'n_iter': 2000,         # Training iterations (default: 1000)
    'n_layers_hidden': 2,   # Hidden layers (default: 1)
    'n_units_hidden': 200,  # Units per layer (default: 100)
    'lr': 0.0005,           # Learning rate (default: 0.001)
    'batch_size': 128,      # Batch size (default: 200)
})
```

### New Model Hyperparameters

```python
# GaussianCopula - SDV statistical copula model
synth = TableSynthesizer('GaussianCopula', config={
    'enforce_min_max_values': True,   # Enforce training data min/max
    'enforce_rounding': True,         # Apply rounding to numerics
    'default_distribution': 'beta',   # Default marginal distribution
})

# TabDiff - Diffusion-style statistical synthesizer (synthcity backend)
synth = TableSynthesizer('TabDiff', config={
    'noise_scale': 0.05,              # Gaussian noise added to samples
    'covariance_regularization': 1e-5, # Ridge regularization for covariance
    'random_state': 42,               # Reproducibility seed
})

# TabPFGen - SGLD with optional TabPFN refinement
synth = TableSynthesizer('TabPFGen', config={
    'n_sgld_steps': 25,               # Langevin dynamics iterations
    'sgld_step_size': 0.01,           # SGLD step size
    'sgld_noise_scale': 0.01,         # Noise in Langevin dynamics
    'use_tabpfn_refinement': True,    # Use TabPFN for target prediction
    'classification_max_unique': 20,  # Threshold for classification vs regression
})
```

### Common Commands

```bash
# Quick test
python tests/integration/test_models_comprehensive.py --mode ultra-quick

# Train a model
python train_all_compatible_models.py --dataset insurance --models CTGAN --epochs 50

# Run tests
pytest tests/ -v

# Check GPU
python -c "from stg.gpu_utils import print_gpu_info; print_gpu_info()"
```

### Quick Start Code

```python
import pandas as pd
from stg.tableSynthesizer import TableSynthesizer

df = pd.read_csv('data.csv')

synth = TableSynthesizer('TVAE', config={'epochs': 100, 'batch_size': 32})
synth.fit(df)
synthetic = synth.sample(n=1000, return_dataframe=True)
```

### Privacy-Preserving Models

```python
# Differential privacy with AIM
synth = TableSynthesizer('AIM', config={'epsilon': 1.0})

# PATE framework with PATECTGAN
synth = TableSynthesizer('PATECTGAN', config={'epsilon': 3.0, 'epochs': 100})

# DP decision tree
synth = TableSynthesizer('DPCART', config={'epsilon': 1.0})
```

### Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Quick testing | Identity | Instant, no training |
| General purpose | TVAE | Fast, good quality |
| Large datasets | CTGAN | Handles scale well |
| High quality | TabDDPM | Best generation quality |
| Privacy required | AIM or PATECTGAN | Differential privacy |
| Imbalanced data | SMOTE | Designed for imbalance |
| Interpretability | CART | Decision tree based |
| Causal modeling | BayesianNetwork | Learns dependencies |
| Complex distributions | AutoDiff | VAE + Diffusion hybrid |
