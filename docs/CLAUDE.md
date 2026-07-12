# Development Guide

This page is the detailed development reference; it expands on the root `CLAUDE.md`
(the project's quick-start guide for Claude Code), viewable on GitHub at
[CLAUDE.md](https://github.com/UCLA-Trustworthy-AI-Lab/table-synthesizers/blob/main/CLAUDE.md).

## Table of Contents

- [Project Architecture](#project-architecture)
- [BaseSynthesizer API Reference](#basesynthesizer-api-reference)
- [Model Implementation Guide](#model-implementation-guide)
- [Configuration System](#configuration-system)
- [Manager Infrastructure](#manager-infrastructure)
- [GPU Utilities](#gpu-utilities)
- [Testing Guide](#testing-guide)
- [Dependency Management](#dependency-management)
- [Troubleshooting](#troubleshooting)

## Project Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Training Scripts                        │
│  train_all_compatible_models.py                         │
│  (single dataset / iterative / batch training)          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              TableSynthesizer (Factory)                  │
│  src/stg/tableSynthesizer.py                            │
│  Maps model names to classes, handles instantiation     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              BaseSynthesizer (Abstract Base)             │
│  src/stg/base.py                                        │
│  Common interface: train(), generate(), set_device()    │
│  Managers: DataManager, ConfigManager                   │
└──────────────┬───────────────────────┬──────────────────┘
               │                       │
   ┌───────────▼────────┐  ┌──────────▼───────────┐
   │  Custom Models      │  │  Synthcity Models    │
   │  CTGAN, TVAE,       │  │  ARF, BayesianNet,   │
   │  TabDDPM, TabSyn,   │  │  GREAT, NFlow        │
   │  AutoDiff, PATECTGAN│  │  (plugin wrappers)   │
   │  CART, DPCART, SMOTE│  │                      │
   │  AIM, Identity,     │  │                      │
   │  LTM_VAE            │  │                      │
   └─────────────────────┘  └──────────────────────┘
```

### Data Flow

1. **Input**: Raw `pd.DataFrame` (mixed types: numerical, categorical, binary)
2. **Preprocessing**: Each model handles its own encoding internally
   - Numerical: MinMaxScaler, StandardScaler
   - Categorical: OneHotEncoder, LabelEncoder
   - The caller does NOT need to preprocess
3. **Training**: Model-specific training loop (GAN, VAE, diffusion, etc.)
4. **Generation**: Model produces synthetic samples
5. **Output**: Returns `torch.Tensor` or `pd.DataFrame` (via `return_dataframe=True`)

### File Organization

```
src/stg/
├── base.py                    # BaseSynthesizer abstract class (770+ lines)
│                              #   - train(), generate(), set_device()
│                              #   - get_optimal_batch_size(), get_state(), load_state()
│                              #   - Threading support for progress reporting
├── tableSynthesizer.py        # Factory class mapping names to model classes
├── config_manager.py          # JSON config loading and validation
├── data_manager.py            # Unified storage (checkpoints, models, outputs)
├── metrics_manager.py         # Training metrics collection
├── wandb_manager.py           # WandB experiment tracking integration
├── gpu_utils.py               # GPU detection, batch size optimization
├── zero_workaround.py         # libzero replacement for TabDDPM/TabSyn
├── path_config.py             # Path resolution utilities
│
├── identity/                  # Identity (baseline, returns training data)
├── TVAE/                      # Tabular VAE
├── CTGAN/                     # Conditional GAN
│   ├── ctgan.py               # Main CTGAN implementation
│   ├── models.py              # Generator and Discriminator networks
│   └── data_sampler.py        # Training data sampling
├── PATECTGAN/                 # Privacy-aware CTGAN + PATE
├── TabDDPM/                   # Diffusion model
│   └── tab_ddpm/              # Core diffusion implementation
│       ├── utils.py           # Utilities and data handling
│       ├── train.py           # Training loop (subprocess-based)
│       └── sample.py          # Sampling loop (subprocess-based)
├── TabSyn/                    # Advanced tabular synthesis
│   ├── tabsyn_synthesizer.py  # Main synthesizer wrapper
│   ├── tabsyn/                # Core TabSyn model
│   ├── src/                   # TabSyn source modules
│   └── utils_train.py         # Training utilities
├── AutoDiff/                  # VAE + Diffusion hybrid
├── LTM_VAE.py                 # Latent Table Model wrapper (single file)
├── SMOTE/                     # SMOTE oversampling
├── CART/                      # Decision tree synthesis
├── DPCART/                    # DP decision tree
├── AIM/                       # Adaptive Iterative Mechanism
├── BayesianNetwork/           # Bayesian network (synthcity wrapper)
├── ARF/                       # Adversarial Random Forest (synthcity wrapper)
├── NFlow/                     # Normalizing flows (synthcity wrapper)
└── GREAT/                     # Transformer-based (synthcity wrapper)
    ├── __init__.py            # Exports GREATSynthesizer
    └── great_synthesizer.py   # Synthcity plugin wrapper
```

## BaseSynthesizer API Reference

### Constructor

```python
class BaseSynthesizer:
    def __init__(
        self,
        data_info=None,                    # Data transformation info
        checkpoint_interval_seconds=None,  # Checkpoint frequency
        epochs=None,                       # Training epochs
        messageSender=None,                # Progress reporting
        seed: int = None,                  # Random seed
        enable_data_manager: bool = True,  # Enable DataManager
        enable_config_manager: bool = True,# Enable ConfigManager
        data_dir: Optional[str] = None,    # DataManager base dir
        config_dir: Optional[str] = None,  # ConfigManager config dir
        **kwargs
    )
```

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `train()` | `train(train_data, batch_size=32)` | Train on DataFrame or DataLoader |
| `generate()` | `generate(n)` | Generate n synthetic samples |
| `set_device()` | `set_device(device=None)` | Set compute device (auto/cuda/mps/cpu) |
| `get_optimal_batch_size()` | `get_optimal_batch_size(dataset_size, default=128)` | GPU-aware batch sizing |
| `init_model()` | `init_model(train_data)` | Initialize model attributes from data |
| `get_state()` | `get_state()` | Serialize model state |
| `load_state()` | `load_state(checkpoint)` | Restore model from checkpoint |
| `set_seed()` | `set_seed(seed)` | Set random seed |

### Methods Subclasses Must Override

```python
def _train(self, train_data):
    """Internal training implementation. Called by train()."""
    raise NotImplementedError

def _generate(self, n):
    """Internal generation implementation. Called by generate()."""
    raise NotImplementedError
```

### Optional Overrides

```python
def init_model(self, train_data):
    """Initialize model-specific attributes from training data."""
    pass

def get_state(self):
    """Return dict of model state for checkpointing."""
    return {}

def load_state(self, checkpoint):
    """Restore model from checkpoint dict."""
    pass
```

## Model Implementation Guide

### Step 1: Create Model Directory

```bash
mkdir src/stg/MyModel
touch src/stg/MyModel/__init__.py
```

### Step 2: Implement Synthesizer

```python
# src/stg/MyModel/__init__.py
import torch
import pandas as pd
from ..base import BaseSynthesizer

class MyModelSynthesizer(BaseSynthesizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_param = kwargs.get('my_param', 0.5)

    def _train(self, train_data):
        """
        Args:
            train_data: pd.DataFrame with raw data
        """
        # 1. Encode the data (handle categorical/numerical)
        # 2. Build model architecture
        # 3. Training loop
        # 4. Store trained state
        pass

    def _generate(self, n):
        """
        Args:
            n: Number of samples to generate

        Returns:
            pd.DataFrame or torch.Tensor
        """
        # 1. Sample from the model
        # 2. Decode back to original format
        # 3. Return as DataFrame
        pass

    def get_state(self):
        return {
            'model_state': self.model.state_dict(),
            'my_param': self.my_param,
        }

    def load_state(self, checkpoint):
        self.my_param = checkpoint.get('my_param', 0.5)
        self.model.load_state_dict(checkpoint['model_state'])
```

### Step 3: Register in Factory

```python
# In src/stg/tableSynthesizer.py, add:
try:
    from .MyModel import MyModelSynthesizer
    MYMODEL_AVAILABLE = True
except ImportError:
    MYMODEL_AVAILABLE = False

# In DEFAULT_MODELS or conditional block:
if MYMODEL_AVAILABLE:
    DEFAULT_MODELS["MyModel"] = MyModelSynthesizer
```

### Step 4: Add Configuration

Create `config/default_MyModel.json`:

```json
{
    "model_type": "gpu",
    "epochs": 50,
    "batch_size": 128,
    "my_param": 0.5,
    "description": "My custom synthesizer",
    "speedup": "10-20x",
    "quality": "Good"
}
```

### Step 5: Add Tests

```python
# tests/unit/test_mymodel.py
import pytest
import pandas as pd
from stg.MyModel import MyModelSynthesizer

def test_mymodel_init():
    synth = MyModelSynthesizer(epochs=1)
    assert synth._epochs == 1

def test_mymodel_train_and_generate():
    df = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0],
        'b': ['x', 'y', 'x', 'y', 'x']
    })
    synth = MyModelSynthesizer(epochs=1)
    synth.train(df)
    result = synth.generate(10)
    assert len(result) == 10
```

## Configuration System

### ConfigManager

The `ConfigManager` (`src/stg/config_manager.py`) loads JSON configs:

```python
from stg.config_manager import ConfigManager

cm = ConfigManager(config_dir='config')
config = cm.load_config('CTGAN')  # Loads config/default_CTGAN.json
```

### Config File Format

```json
{
    "model_type": "gpu",
    "epochs": 50,
    "batch_size": 200,
    "embedding_dim": 128,
    "generator_dim": [256, 256],
    "discriminator_dim": [256, 256],
    "description": "Conditional GAN",
    "speedup": "15-30x",
    "quality": "High"
}
```

### Auto-Conversion Rules

- JSON arrays for dimension parameters are auto-converted to Python tuples
- Affected fields: `generator_dim`, `discriminator_dim`, `compress_dims`, `decompress_dims`
- Example: `[256, 256]` in JSON becomes `(256, 256)` in Python

### Runtime Overrides

Command-line arguments in `train_all_compatible_models.py` override config file values. Priority: CLI args > config file > model defaults.

### Synthcity Plugin Parameter Passthrough

Synthcity-based models (GREAT, BayesianNetwork, ARF, NFlow) use a different config pattern than custom models. They extract plugin-specific parameters from `**kwargs` in `__init__()` via a `_SYNTHCITY_PARAMS` set, then pass them to `Plugins().get("plugin_name", **kwargs)` in `train()`.

Config metadata fields (`model_type`, `description`, `speedup`, `quality`, `backend`) are automatically stripped by `get_model_config()` and never reach the synthesizer.

**GREAT parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iter` | 100 | Training epochs (range: 50-500) |
| `llm` | distilgpt2 | HuggingFace LLM model name |
| `batch_size` | 8 | Training batch size |
| `device` | cpu | `cpu` or `cuda` |
| `random_state` | 0 | Random seed |
| `sampling_patience` | 500 | Max schema-valid retries |
| `logging_epoch` | 100 | Log every N epochs |

**BayesianNetwork parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `struct_learning_n_iter` | 1000 | Structure learning iterations |
| `struct_learning_search_method` | tree_search | `hillclimb`, `pc`, `tree_search` |
| `struct_learning_score` | k2 | `bdeu`, `bds`, `bic`, `k2` |
| `struct_max_indegree` | 4 | Max parent nodes per variable |
| `encoder_max_clusters` | 10 | Discretization clusters |
| `encoder_noise_scale` | 0.1 | Anti-leakage noise |
| `random_state` | 0 | Random seed |
| `sampling_patience` | 500 | Max schema-valid retries |

**ARF parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_trees` | 30 | Number of trees |
| `delta` | 0 | Convergence threshold |
| `max_iters` | 10 | Max training iterations |
| `early_stop` | true | Enable early stopping |
| `min_node_size` | 5 | Min leaf node size |
| `random_state` | 0 | Random seed |
| `sampling_patience` | 500 | Max schema-valid retries |

**NFlow parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iter` | 1000 | Training iterations |
| `n_layers_hidden` | 1 | Hidden layers |
| `n_units_hidden` | 100 | Units per hidden layer |
| `batch_size` | 200 | Training batch size |
| `num_transform_blocks` | 1 | Transform blocks |
| `dropout` | 0.1 | Dropout rate |
| `batch_norm` | false | Batch normalization |
| `num_bins` | 8 | Spline bins |
| `lr` | 0.001 | Learning rate |
| `encoder_max_clusters` | 10 | Encoding clusters |
| `patience` | 5 | Early stopping patience |
| `device` | cpu | `cpu` or `cuda` |
| `random_state` | 0 | Random seed |
| `sampling_patience` | 500 | Max schema-valid retries |

Note: For GREAT and NFlow, the `epochs` CLI argument maps to `n_iter` automatically.

## Manager Infrastructure

### DataManager (`src/stg/data_manager.py`)

Handles unified storage for all model artifacts:

```python
from stg.data_manager import DataManager

dm = DataManager(base_dir='data')
dm.save_model(model_state, 'CTGAN', 'insurance')
dm.save_checkpoint(checkpoint, 'CTGAN', 'insurance', epoch=50)
dm.save_output(synthetic_df, 'CTGAN', 'insurance')
```

### MetricsManager (`src/stg/metrics_manager.py`)

Collects training metrics:

```python
from stg.metrics_manager import MetricsManager

mm = MetricsManager()
mm.log_metric('train_loss', 0.5, step=100)
mm.log_metric('val_loss', 0.6, step=100)
```

### WandBManager (`src/stg/wandb_manager.py`)

Integrates with Weights & Biases:

```python
from stg.wandb_manager import WandBManager

wm = WandBManager(project='my-project')
wm.init(config={'epochs': 50, 'model': 'CTGAN'})
wm.log({'loss': 0.5}, step=100)
wm.finish()
```

## GPU Utilities

### `src/stg/gpu_utils.py` Functions

| Function | Description |
|----------|-------------|
| `detect_best_device()` | Auto-detect best device (CUDA > MPS > CPU) |
| `get_device_info()` | Detailed device info dict |
| `is_gpu_available()` | Boolean GPU check |
| `get_optimal_batch_size(dataset_size)` | Memory-aware batch sizing |
| `get_gpu_models_supported()` | Model GPU support matrix |
| `print_gpu_info()` | Pretty-print GPU info |
| `validate_gpu_setup()` | Validate GPU configuration |

### Batch Size Optimization

The `get_optimal_batch_size()` function uses GPU memory tiers:

| GPU Memory | Recommended Batch Size |
|-----------|----------------------|
| 80+ GB | 1024 |
| 40+ GB | 512 |
| 16+ GB | 256 |
| 8+ GB | 128 |
| < 8 GB | 64 |
| CPU/MPS | 64 (conservative) |

## Testing Guide

### Test Structure

```
tests/
├── conftest.py                         # Global fixtures (sample DataFrames)
├── unit/
│   ├── conftest.py                     # Unit test fixtures
│   ├── test_aim.py                     # AIM unit tests
│   ├── test_arf.py                     # ARF unit tests
│   ├── test_autodiff.py                # AutoDiff unit tests
│   ├── test_bayesian_network.py        # BayesianNetwork unit tests
│   ├── test_cart.py                    # CART unit tests
│   ├── test_ctgan.py                   # CTGAN unit tests
│   ├── test_dpcart.py                  # DPCART unit tests
│   ├── test_great.py                   # GREAT unit tests
│   ├── test_ltm_vae.py                # LTM_VAE unit tests
│   ├── test_nflow.py                   # NFlow unit tests
│   ├── test_patectgan.py              # PATECTGAN unit tests
│   ├── test_smote.py                   # SMOTE unit tests
│   ├── test_tabddpm.py                # TabDDPM unit tests
│   ├── test_tabsyn.py                 # TabSyn unit tests
│   └── test_tvae.py                    # TVAE unit tests
└── integration/
    ├── utils.py                        # Shared test utilities
    ├── dataframe_test_utils.py         # DataFrame comparison helpers
    ├── test_models_comprehensive.py    # All-model test runner
    ├── test_{model}_integration.py     # Per-model integration tests
    └── test_data/                      # Test datasets
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Single model
pytest tests/unit/test_ctgan.py -v
pytest tests/integration/test_ctgan_integration.py -v

# By category
./run_comprehensive_tests.sh --core
./run_comprehensive_tests.sh --stable
./run_comprehensive_tests.sh --experimental

# Quick validation (all models, minimal epochs)
python tests/integration/test_models_comprehensive.py --mode ultra-quick

# Specific model quick test
python tests/integration/test_models_comprehensive.py --model TVAE --mode quick
```

### Writing Tests

Unit test pattern:

```python
import pytest
import pandas as pd

class TestMyModel:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'numerical': [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            'categorical': ['a', 'b', 'c', 'a', 'b'] * 20,
        })

    def test_init(self):
        from stg.MyModel import MyModelSynthesizer
        synth = MyModelSynthesizer(epochs=1)
        assert synth is not None

    def test_train(self, sample_data):
        from stg.MyModel import MyModelSynthesizer
        synth = MyModelSynthesizer(epochs=1, batch_size=16)
        synth.train(sample_data)

    def test_generate(self, sample_data):
        from stg.MyModel import MyModelSynthesizer
        synth = MyModelSynthesizer(epochs=1, batch_size=16)
        synth.train(sample_data)
        result = synth.generate(50)
        assert len(result) == 50
```

## Dependency Management

### 4-Tier Requirements Structure

Dependencies are split into four files so users install only what they need:

#### `requirements.txt` (Base)

Core packages for CPU-only models (Identity, CART, DPCART, SMOTE, AIM). No torch.

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0.0 | DataFrame handling |
| numpy | >= 1.26.4, < 2.0.0 | Numerical computing |
| scikit-learn | >= 1.7.2 | ML utilities, preprocessing |
| scipy | >= 1.11.4 | Scientific computing |
| imbalanced-learn | >= 0.14.1 | SMOTE implementation |
| category_encoders | >= 2.6.0 | Categorical encoding |
| tqdm | >= 4.60.0 | Progress bars |
| tomli / tomli_w | >= 2.0.0 / >= 1.2.0 | TOML parsing (TabSyn) |
| icecream | >= 2.1.0 | Debug output (TabSyn) |
| seaborn | latest | Visualization |
| pytest | >= 7.0.0 | Testing |

#### `requirements-gpu.txt` (GPU - CUDA SM 12.1 Blackwell)

For GPU-accelerated models (CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, TabSyn, LTM_VAE).

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.7.0 | PyTorch with CUDA support |
| torchvision | >= 0.22.0 | Vision utilities |
| transformers | >= 4.0.0 | LTM / text embeddings |
| datasets | >= 2.0.0 | HuggingFace datasets |
| accelerate | >= 0.20.0 | Training acceleration |
| ema-pytorch | >= 0.7.0 | Exponential moving average |
| boto3 | >= 1.26.0 | S3 storage backend |

**Install torch with CUDA first:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### `requirements-cpu.txt` (CPU torch)

Same models as GPU tier but running on CPU. Identical packages, different torch wheel.

**Install torch CPU-only first:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### `requirements-synthcity.txt` (Synthcity)

For BayesianNetwork, ARF, GREAT, NFlow. Best on Python 3.11 (ts311 conda env).

| Package | Version | Purpose |
|---------|---------|---------|
| synthcity | >= 0.2.12 | All 4 synthcity plugin models |

**Known issues:** Python 3.12 + ARM64 may lack torchtext wheels.

### Installation Patterns

```bash
# Minimal: CPU-only models (no torch)
pip install -r requirements.txt

# GPU models on Blackwell (SM 12.1, CUDA 13.0+)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-gpu.txt

# GPU models on older NVIDIA (Ampere/Hopper, CUDA 12.1)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt

# CPU torch models (no GPU needed)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt

# Synthcity models (add to any of the above)
pip install -r requirements-synthcity.txt
```

### Model-to-Tier Mapping

| Tier | Models |
|------|--------|
| **Base** | Identity, CART, DPCART, SMOTE, AIM |
| **GPU / CPU** | CTGAN, TVAE, TabDDPM, PATECTGAN, AutoDiff, TabSyn, LTM_VAE |
| **Synthcity** | BayesianNetwork, ARF, GREAT, NFlow |

### The libzero Workaround

TabDDPM and TabSyn originally depend on `libzero`. We use `zero_workaround.py` as a drop-in replacement:

```python
# In TabDDPM/TabSyn code, instead of:
# import zero
# The code imports from zero_workaround which provides the same API
```

This avoids the `libzero` package which has compatibility issues.

## Troubleshooting

### Common Import Errors

```python
# ModuleNotFoundError: No module named 'stg'
# Solution: Ensure src/ is in sys.path
import sys; sys.path.insert(0, 'src')

# Or install the package
pip install -e .
```

### "Model X not available, skipping"

This means the model's import failed silently at startup. The training script now shows
actionable diagnostics:

```
Warning: Model GreaT (maps to GREAT) not available, skipping
  Hint: Install synthcity: pip install -r requirements-synthcity.txt
       Import error: No module named 'synthcity'
```

**Common causes:**
- Synthcity models (GREAT, ARF, BayesianNetwork, NFlow): synthcity not installed
- GPU models (CTGAN, TVAE, TabDDPM, etc.): torch not installed
- Wrong conda environment (synthcity models need `ts311` env)

**Debug manually:**
```python
# Check which models are loaded
python -c "from stg.tableSynthesizer import DEFAULT_MODELS; print(list(DEFAULT_MODELS.keys()))"

# Test a specific model's import
python -c "from stg.GREAT import GREATSynthesizer; print('OK')"
```

### TabSyn + Python 3.12

TabSyn uses subprocess-based training that can fail on Python 3.12:
- `pkgutil.ImpImporter` was removed in Python 3.12
- Workaround: Use Python 3.10/3.11, or use TabDDPM as an alternative

### synthcity + ARM64 Linux

synthcity requires `torchtext` which lacks ARM64 wheels:
- Affected models: GREAT, NFlow (and potentially ARF, BayesianNetwork)
- Workaround: Use x86_64 or use alternative models (CTGAN, TabDDPM)

### GREAT + transformers 5.x

be_great 0.0.9 uses `tokenizer=` in HuggingFace Trainer, but transformers 5.0 renamed it
to `processing_class=`. The `_patch_great_trainer()` monkey-patch in `great_synthesizer.py`
handles this automatically. If you see signature errors from transformers, check that the
patch is loading (set `logging.basicConfig(level=logging.INFO)`).

### CUDA Out of Memory

```python
# Reduce batch size
config = {'epochs': 50, 'batch_size': 32}

# Or use automatic optimization
from stg.gpu_utils import get_optimal_batch_size
batch_size = get_optimal_batch_size(len(df))
```

### Zero Module Errors

If you see errors about `zero` module:
- Verify `zero_workaround.py` exists in `src/stg/`
- Check that TabDDPM/TabSyn imports reference the workaround, not the original `libzero`

## Platform Compatibility Matrix

| Model | Python 3.10 | Python 3.11 | Python 3.12 | CUDA | MPS | CPU |
|-------|:-----------:|:-----------:|:-----------:|:----:|:---:|:---:|
| Identity | Y | Y | Y | - | - | Y |
| TVAE | Y | Y | Y | Y | Y | Y |
| CTGAN | Y | Y | Y | Y | Y | Y |
| PATECTGAN | Y | Y | Y | Y | Y | Y |
| TabDDPM | Y | Y | Y | Y | Y | Y |
| AutoDiff | Y | Y | Y | Y | Y | Y |
| TabSyn | Y | Y | * | Y | Y | Y |
| LTM_VAE | Y | Y | Y | Y | Y | Y |
| CART | Y | Y | Y | - | - | Y |
| DPCART | Y | Y | Y | - | - | Y |
| SMOTE | Y | Y | Y | - | - | Y |
| AIM | Y | Y | Y | - | - | Y |
| ARF | Y | Y | ** | - | - | Y |
| BayesianNetwork | Y | Y | ** | - | - | Y |
| GREAT | Y | Y | ** | Y | - | Y |
| NFlow | Y | Y | ** | Y | - | Y |

`*` TabSyn has subprocess compatibility issues on Python 3.12
`**` synthcity models may have dependency issues on Python 3.12 + ARM64
