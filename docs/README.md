# Documentation Index

Complete documentation for the table-synthesizers project. For a quick overview, see the root [README.md](../README.md).

## Getting Started

| Document | Description |
|----------|-------------|
| [Root README](../README.md) | Project overview, installation, and quick start |
| [CLAUDE.md (root)](../CLAUDE.md) | Development guide and project constraints |
| [Development Guide](CLAUDE.md) | Detailed architecture, API reference, and contributing guide |
| [Config System](../config/README.md) | JSON configuration files and customization |

## Model Comparisons & References

| Document | Description |
|----------|-------------|
| [Algorithm Comparison Summary](ALGORITHM_COMPARISON_SUMMARY.txt) | CTGAN vs PATE-CTGAN quick comparison |
| [Hyperparameter Comparison Summary](HYPERPARAMETER_COMPARISON_SUMMARY.txt) | Hyperparameter tuning reference across models |

## API Reference (mkdocs site)

| Document | Description |
|----------|-------------|
| [BaseSynthesizer](api/base.md) | Abstract base class all synthesizers implement |
| [TableSynthesizer](api/table_synthesizer.md) | Factory/unified interface for all registered models |

## Quick Reference

### Available Models (16 total)

**GPU-Accelerated (10-50x speedup)**:
CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, GREAT, NFlow, TabSyn, LTM_VAE

**CPU-Only**:
Identity, CART, DPCART, SMOTE, AIM, BayesianNetwork, ARF

### Installation (4-Tier Requirements)

```bash
# Base only (Identity, CART, DPCART, SMOTE, AIM - no torch needed)
pip install -r requirements.txt

# GPU models on Blackwell (CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, TabSyn)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-gpu.txt

# CPU torch models (same GPU models, running on CPU)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt

# Synthcity models (BayesianNetwork, ARF, GREAT, NFlow - best on Python 3.11)
pip install -r requirements-synthcity.txt
```

| File | Provides | Key Packages |
|------|----------|-------------|
| `requirements.txt` | Base (CPU-only models) | pandas, numpy, sklearn, scipy |
| `requirements-gpu.txt` | GPU models (CUDA SM 12.1) | torch>=2.7, torchvision, transformers |
| `requirements-cpu.txt` | GPU models on CPU | torch>=2.7 (CPU wheel), transformers |
| `requirements-synthcity.txt` | Synthcity models | synthcity>=0.2.12 |

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
