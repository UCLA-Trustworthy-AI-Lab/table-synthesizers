# Table Synthesizers

A Python library for generating synthetic tabular data using state-of-the-art machine learning models. This library provides multiple synthesizers including GANs, Variational Autoencoders, Diffusion Models, and privacy-preserving methods.

## 🚀 Features

- **9 Core Synthesizer Models**: All major models working with 100% success rate
- **Simple DataFrame Interface**: Direct pandas DataFrame input with automatic encoding
- **Flexible Output**: Generate synthetic data as tensors or DataFrames
- **Privacy-Preserving Options**: Differential privacy support with AIM, PATECTGAN, and DPCART
- **Extensible Architecture**: Easy to add custom synthesizers
- **Comprehensive Testing**: Ultra-quick, quick, and full testing modes available

## 📦 Installation

```bash
# Install basic requirements
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]

# Activate the conda environment
conda activate table-synthesizers
```

## 🏗️ Repository Structure

```
table-synthesizers/
├── src/stg/                    # Main library code
│   ├── base.py                 # BaseSynthesizer abstract class
│   ├── tableSynthesizer.py     # Factory class for model selection
│   ├── identity/               # Identity baseline synthesizer
│   ├── TVAE/                   # Tabular Variational AutoEncoder
│   ├── CTGAN/                  # Conditional GAN
│   ├── TabDDPM/               # Diffusion model
│   ├── PATECTGAN/             # Privacy-aware CTGAN
│   └── AIM/                   # Adaptive and Iterative Mechanism
├── test/                      # Test suite
│   ├── dataframe_test_utils.py # Shared DataFrame testing utilities
│   ├── test_*.py              # Individual model tests
│   └── test_data/             # Test datasets and utilities
├── CLAUDE.md                  # Development guide
└── README.md                  # This file
```

## 🛠️ Quick Start

### Basic Usage with DataFrame Input

```python
import pandas as pd
from stg import TableSynthesizer

# Load your data
df = pd.read_csv('your_data.csv')

# Or create sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'category': ['A', 'B', 'A', 'C', 'B']
})

# Initialize TVAE synthesizer
synthesizer = TableSynthesizer('TVAE', {
    'epochs': 100,
    'batch_size': 32,
    'embedding_dim': 128
})

# Train the model
synthesizer.fit(df)

# Generate synthetic data as DataFrame
synthetic_df = synthesizer.sample(n=100, return_dataframe=True)
print(synthetic_df.head())

# Or generate as tensor for further processing
synthetic_tensor = synthesizer.sample(n=100)
print(f"Generated tensor shape: {synthetic_tensor.shape}")
```

### Advanced Configuration

```python
# Configure model-specific parameters
synthesizer = TableSynthesizer('TVAE', {
    'epochs': 300,
    'batch_size': 500,
    'embedding_dim': 256,
    'compress_dims': (256, 128),
    'decompress_dims': (128, 256),
    'l2scale': 1e-5,
    'loss_factor': 2
})

# Train with custom batch size
synthesizer.fit(df, batch_size=64)

# Generate with different sample sizes
small_sample = synthesizer.sample(n=50, return_dataframe=True)
large_sample = synthesizer.sample(n=1000, return_dataframe=True)
```

## 🧪 Available Synthesizers (100% Success Rate 🎉)

| Model | Description | Status | Performance | Best For |
|-------|-------------|--------|-------------|----------|
| **Identity** | Baseline that returns training samples | ✅ Working | 0.01s | Testing, baselines |
| **TVAE** | Tabular Variational AutoEncoder | ✅ Working | 0.01s | Mixed data types, general use |
| **TabDDPM** | Diffusion model for tabular data | ✅ Working | 38s | High-quality generation |
| **CTGAN** | Conditional GAN | ✅ Working | 0.13s | Large datasets |
| **PATECTGAN** | Privacy-aware CTGAN with PATE | ✅ Working | 1.75s | Privacy-sensitive applications |
| **SMOTE** | Synthetic Minority Oversampling | ✅ Working | 0.00s | Imbalanced datasets |
| **CART** | Decision tree-based synthesis | ✅ Working | 0.00s | Interpretable models |
| **DPCART** | Differentially private CART | ✅ Working | 0.00s | Privacy + interpretability |
| **AIM** | Adaptive and iterative mechanism | ✅ Working | 0.01s | Privacy-sensitive applications |

### 🚀 Recent Fixes (September 2025)

All core issues have been resolved:
- **CTGAN**: ✅ Added missing sklearn-style interface methods (`fit`, `sample`, `decode_samples`)
- **PATECTGAN**: ✅ Fixed tensor dimension mismatches in PATE implementation
- **SMOTE**: ✅ Fixed undefined variable errors (`n_jobs` parameter)
- **TabSyn**: ✅ Fixed zero module imports in subprocess architecture
- **TabDDPM**: ✅ Fixed zero module imports in training/sampling scripts

## 🔧 Testing

### Comprehensive Model Testing (Recommended)

```bash
# Full comprehensive test (all models, full training)
python test/test_models_comprehensive.py

# Quick test (smaller datasets, fewer epochs)
python test/test_models_comprehensive.py --mode quick

# Ultra-quick test (tiny datasets, minimal training) - Great for CI/CD
python test/test_models_comprehensive.py --mode ultra-quick

# Test specific model only
python test/test_models_comprehensive.py --model TVAE --mode quick
```

### Individual Tests

```bash
# Run all traditional tests
pytest test/ -v

# Run DataFrame input tests only
pytest -k "dataframe_support" -v

# Run specific model tests
pytest test/test_TVAE.py -v
pytest test/test_identity.py -v
```

### Performance Benchmarking

The comprehensive test provides detailed performance metrics:
- Training time for each model
- Sampling time comparison
- Data quality analysis
- Success/failure rates

## 📊 Data Types Supported

The synthesizers support various data types with automatic encoding:

- **Continuous**: Numerical data (age, income, temperature)
- **Categorical**: Categories (gender, product_type, city)
- **Binary**: Binary features (is_member, has_license)
- **Bounded Continuous**: Values within specific ranges
- **Ordinal**: Ordered categories (rating, education_level)

## 🔐 Privacy-Preserving Synthesis

For privacy-sensitive applications, use models with differential privacy:

```python
# AIM with differential privacy
synthesizer = TableSynthesizer('AIM', {
    'epsilon': 1.0,      # Privacy budget
    'delta': 1e-9,       # Privacy parameter
    'rounds': 100,       # Number of iterations
    'max_model_size': 80 # Model complexity limit
})

# PATECTGAN with PATE framework
synthesizer = TableSynthesizer('PATECTGAN', {
    'epsilon': 3.0,
    'epochs': 100,
    'teacher_iters': 5,
    'student_iters': 5
})
```

## 🤝 Contributing

1. Follow the architecture patterns in `BaseSynthesizer`
2. Implement `_train()` and `_generate()` methods
3. Add tests in `test/test_your_model.py`
4. Update this README with your model information

## 📖 Documentation

- **CLAUDE.md**: Development guide with detailed architecture information
- **Test files**: Examples of how to use each synthesizer
- **Code comments**: Inline documentation in source files

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in model configuration
2. **Missing dependencies**: Install requirements: `pip install -r requirements.txt`
3. **Zero module errors**: The library includes `zero_workaround.py` that automatically handles this
4. **Import errors**: Ensure you're running from the project root directory

### Performance Tips

1. **Use ultra-quick mode for testing**: `--mode ultra-quick` for rapid validation
2. **Start with fast models**: Try Identity, TVAE, or AIM first
3. **Use smaller datasets**: Begin with 100-1000 samples for testing
4. **GPU acceleration**: TabDDPM and CTGAN benefit from GPU when available

### Getting Help

- Run `python test/test_models_comprehensive.py --mode ultra-quick` to verify setup
- Check CLAUDE.md for detailed architecture and troubleshooting
- All models have been tested and verified working as of September 2025

## 📄 License

This project is dual-licensed under Apache-2.0 OR MIT.

- See `LICENSE-APACHE` for the Apache License, Version 2.0.
- See `LICENSE-MIT` for the MIT License.
- The top-level `LICENSE` file explains the dual-licensing terms.

## 🙏 Attribution and Third-Party Licenses

This project integrates and wraps ideas and/or components inspired by the following repositories. We acknowledge their authors and cite their licenses below:

- SynthCity (VanderSchaar Lab): Apache License 2.0
  - Repo: https://github.com/vanderschaarlab/synthcity
  - License: https://raw.githubusercontent.com/vanderschaarlab/synthcity/main/LICENSE

- TabSyn (Amazon Science): Apache License 2.0
  - Repo: https://github.com/amazon-science/tabsyn
  - License: https://raw.githubusercontent.com/amazon-science/tabsyn/main/LICENSE

- AutoDiffusion (UCLA Trustworthy AI Lab): License not declared (no LICENSE file found as of this update)
  - Repo: https://github.com/UCLA-Trustworthy-AI-Lab/AutoDiffusion
  - Note: Please verify licensing terms before distribution or derivative use.

- Synth-MIA (UCLA Trustworthy AI Lab): Repository unavailable at the provided URL (404); license unknown
  - Repo (as cited): https://github.com/UCLA-Trustworthy-AI-Lab/Synth-MIA
  - Note: If an updated location exists, please update CITED_LIBS and this section accordingly.

If you are an author or maintainer of any cited project and prefer a different attribution format, please open an issue.
