# CLAUDE.md

Development guide for Claude Code when working with the table-synthesizers project.

## Overview

**table-synthesizers** is a Python library providing 16 synthetic tabular data generators. All models inherit from `BaseSynthesizer` in `src/stg/base.py` and follow a consistent `fit()` / `sample()` interface. The project wraps multiple backends: custom implementations (CTGAN, TVAE, TabDDPM, TabSyn, AutoDiff), scikit-learn-based models (CART, DPCART, SMOTE, AIM), and synthcity plugins (ARF, BayesianNetwork, GREAT, NFlow).

## Critical Constraints

1. **libzero is replaced** - Never add `libzero` as a dependency. Use `zero_workaround.py` instead. TabDDPM and TabSyn import from this workaround module.

2. **synthcity dependency is optional** - The four synthcity-based models (ARF, BayesianNetwork, GREAT, NFlow) are imported with try/except. Core models must work without synthcity installed.

3. **Imports use `stg` not `src.stg`** - After `sys.path.insert(0, 'src')` in scripts, imports are `from stg.tableSynthesizer import TableSynthesizer`. The package name is `stg`, not `src.stg`.

4. **BaseSynthesizer interface** - All models must implement `_train(train_data)` and `_generate(n)`. The public API is `train()` / `generate()` which wrap these with data preprocessing, device management, and threading support.

5. **DataFrame-first interface** - Models accept raw `pd.DataFrame` input. Encoding (one-hot, label, min-max scaling) is handled inside each synthesizer, not by the caller.

6. **Config files live in `config/`** - Each model has `config/default_{MODEL}.json`. The `ConfigManager` loads these. JSON arrays are auto-converted to Python tuples for dimension parameters.

## Quick Directory Reference

```
src/stg/base.py              # BaseSynthesizer - all models inherit from this
src/stg/tableSynthesizer.py  # Factory: TableSynthesizer('CTGAN', config={...})
src/stg/config_manager.py    # Loads JSON configs from config/ directory
src/stg/data_manager.py      # Unified model storage (checkpoints, outputs)
src/stg/gpu_utils.py         # GPU detection: detect_best_device(), get_optimal_batch_size()
src/stg/zero_workaround.py   # Replacement for libzero dependency
config/default_*.json        # Model configuration files (14 files)
train_all_compatible_models.py  # Main training script (single + iterative modes)
tests/unit/                  # Per-model unit tests
tests/integration/           # End-to-end integration tests
```

## Requirements (4-Tier)

```
requirements.txt            # Base: pandas, numpy, sklearn (no torch)
requirements-gpu.txt        # GPU: torch>=2.7 (CUDA SM 12.1), transformers, accelerate
requirements-cpu.txt        # CPU: torch>=2.7 (CPU wheel), transformers, accelerate
requirements-synthcity.txt  # Synthcity: synthcity>=0.2.12
```

| Tier | Models | Install |
|------|--------|---------|
| Base | Identity, CART, DPCART, SMOTE, AIM | `pip install -r requirements.txt` |
| GPU | CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, TabSyn, LTM_VAE | `pip install torch --index-url .../cu130 && pip install -r requirements-gpu.txt` |
| CPU | Same as GPU, on CPU | `pip install torch --index-url .../cpu && pip install -r requirements-cpu.txt` |
| Synthcity | BayesianNetwork, ARF, GREAT, NFlow | `pip install -r requirements-synthcity.txt` |

## Common Commands

```bash
# Install base + GPU
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-gpu.txt

# Run all tests
pytest tests/ -v

# Quick model validation
python tests/integration/test_models_comprehensive.py --mode ultra-quick

# Train specific models
python train_all_compatible_models.py --dataset insurance --models CTGAN TVAE --epochs 50

# Train all GPU models
python train_all_compatible_models.py --dataset insurance --group gpu

# Train on multiple datasets
python train_all_compatible_models.py --data_folder ./datasets --iterate_datasets --models CTGAN

# Run comprehensive test suite
./run_comprehensive_tests.sh --core

# Check GPU status
python -c "from stg.gpu_utils import print_gpu_info; print_gpu_info()"
```

## Model Registry

Models are registered in `src/stg/tableSynthesizer.py`:

```python
DEFAULT_MODELS = {
    "Identity": Identity,
    "CTGAN": CTGAN,
    "PATECTGAN": PATECTGAN,
    "TVAE": TVAE,
    "CART": CARTSynthesizer,
    "DPCART": DPCARTSynthesizer,
}
# Optional models added via availability flags:
# TabDDPM, AIM, SMOTE, BayesianNetwork, GREAT, ARF, NFlow, AutoDiff, TabSyn, LTM_VAE
```

## Architecture

### BaseSynthesizer (`src/stg/base.py`)

Key methods:
- `__init__()` - Accepts `data_info`, `epochs`, `seed`, manager flags, `**kwargs`
- `train(train_data, batch_size=32)` - Preprocesses data, calls `_train()`
- `generate(n)` - Calls `_generate()`, post-processes output
- `set_device(device=None)` - Auto-detects CUDA > MPS > CPU
- `get_optimal_batch_size(dataset_size)` - GPU-memory-aware batch sizing
- `get_state()` / `load_state(checkpoint)` - Serialization

### Manager System

- **DataManager** - Unified storage for models, checkpoints, and outputs
- **ConfigManager** - Loads and validates JSON configs from `config/` directory
- **MetricsManager** - Tracks training metrics
- **WandBManager** - Weights & Biases experiment tracking

### Adding a New Model

1. Create `src/stg/YourModel/` directory with `__init__.py`
2. Implement a class inheriting from `BaseSynthesizer`
3. Override `_train(train_data)` and `_generate(n)`
4. Add to `tableSynthesizer.py` imports and `DEFAULT_MODELS`
5. Create `config/default_YourModel.json`
6. Add `tests/unit/test_yourmodel.py` and `tests/integration/test_yourmodel_integration.py`

## Device Handling

```python
# Auto-detection (recommended)
synthesizer.model.set_device('auto')  # CUDA > MPS > CPU

# Explicit
synthesizer.model.set_device('cuda')
synthesizer.model.set_device('cpu')
```

GPU models: CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, GREAT, NFlow, TabSyn, LTM_VAE
CPU-only models: Identity, CART, DPCART, SMOTE, AIM, BayesianNetwork, ARF

## Testing Patterns

- Unit tests go in `tests/unit/test_{model}.py`
- Integration tests go in `tests/integration/test_{model}_integration.py`
- Shared fixtures are in `tests/conftest.py` and `tests/unit/conftest.py`
- DataFrame test utilities are in `tests/integration/dataframe_test_utils.py`
- Test data configs are in `tests/integration/test_data/config/`

## Known Issues

- **TabSyn + Python 3.12**: Subprocess import errors with `pkgutil.ImpImporter`
- **synthcity + ARM64 Linux**: `torchtext` wheels unavailable, blocking GREAT and NFlow
- **Checkpoint pickling**: `SimpleTensorDataset` may fail to pickle in some edge cases

## Git Conventions

- Branch: `LTM_package_test` (current development branch)
- Main branch: `main`
- No Claude attribution in commits
