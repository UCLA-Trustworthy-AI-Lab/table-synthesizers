# Table Synthesizers

A Python library for generating synthetic tabular data using state-of-the-art machine learning models. This library provides multiple synthesizers including GANs, Variational Autoencoders, Diffusion Models, and privacy-preserving methods.

## 🚀 Features

- **Multiple Synthesizer Models**: CTGAN, TVAE, TabDDPM, PATECTGAN, AIM, and Identity baseline
- **Simple DataFrame Interface**: Direct pandas DataFrame input with automatic encoding
- **Flexible Output**: Generate synthetic data as tensors or DataFrames
- **Privacy-Preserving Options**: Differential privacy support with AIM and PATECTGAN
- **Extensible Architecture**: Easy to add custom synthesizers

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

## 🧪 Available Synthesizers

| Model | Description | Status | Best For |
|-------|-------------|--------|----------|
| **Identity** | Baseline that returns training samples | ✅ Stable | Testing, baselines |
| **TVAE** | Tabular Variational AutoEncoder | ✅ Stable | Mixed data types, general use |
| **TabDDPM** | Diffusion model for tabular data | ✅ Stable | High-quality generation |
| **CTGAN** | Conditional GAN | ⚠️ DataFrame input has known issues | Large datasets |
| **PATECTGAN** | Privacy-aware CTGAN | ⚠️ DataFrame input has known issues | Privacy-sensitive applications |
| **AIM** | Differential privacy mechanism | ⚠️ Requires `mbi` package | Privacy-sensitive applications |

### Known Issues

- **CTGAN & PATECTGAN**: DataFrame input currently has dimension mismatch issues with PAC (Packing) parameter. Legacy DataLoader input works correctly.
- **AIM**: Requires the `mbi` package which may not be available via standard package managers.

## 🔧 Testing

```bash
# Run all tests
source ~/anaconda3/etc/profile.d/conda.sh && conda activate table-synthesizers && pytest test/ -v

# Run DataFrame input tests only
source ~/anaconda3/etc/profile.d/conda.sh && conda activate table-synthesizers && pytest -k "dataframe_support" -v

# Run specific model tests
pytest test/test_TVAE.py -v
pytest test/test_identity.py -v
```

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
2. **Dimension mismatch**: Check input data format and model parameters
3. **Missing dependencies**: Install requirements and activate conda environment
4. **AIM not available**: Install the `mbi` package or use alternative models

### Getting Help

- Check existing tests for usage examples
- Review CLAUDE.md for architecture details
- Open an issue for bugs or feature requests

## 📄 License

[Add your license information here]
