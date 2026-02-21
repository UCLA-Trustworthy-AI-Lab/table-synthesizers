# Implementation Plan: Multi-Model Synthetic Data Generation

## Objective
Train multiple table synthesizers (TabSyn, CTGAN, GReat, PATE-CTGAN) on 80% of data and generate 1000 synthetic samples from each model.

## Dataset Information
- **Input Location**: `/home/ohsono/dataset/input_data/`
- **Train/Test Split**: 80/20
- **Target Synthetic Samples**: 1000 rows per model

## Target Models
1. **TabSyn** - VAE + Diffusion approach
2. **CTGAN** - Conditional GAN
3. **GReat** - Transformer-based generator
4. **PATE-CTGAN** - Privacy-preserving GAN

---

## Implementation Steps

### Step 1: Data Preparation
**Goal**: Load data from input folder and split into train/test sets

**Tasks**:
- [ ] Identify data file(s) in `/home/ohsono/dataset/input_data/`
- [ ] Load data into pandas DataFrame
- [ ] Perform 80/20 train/test split with stratification (if applicable)
- [ ] Save split datasets for reproducibility
  - `train_data.csv` (80%)
  - `test_data.csv` (20%)

**Implementation Approach**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data_path = "/home/ohsono/dataset/input_data/your_file.csv"
df = pd.read_csv(data_path)

# Split 80/20
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    # stratify=df['target_column'] if applicable
)

# Save splits
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
```

**Expected Output**:
- `train_data.csv` (80% of original data)
- `test_data.csv` (20% of original data)

---

### Step 2: Model Training - TabSyn
**Goal**: Train TabSyn synthesizer on training data

**Characteristics**:
- Input: DataFrame only
- Approach: VAE + Diffusion subprocess-based
- Training: Spawns external subprocess
- Output: DataFrame (native)

**Tasks**:
- [ ] Initialize TabSyn synthesizer
- [ ] Train on `train_df`
- [ ] Save model checkpoint
- [ ] Generate 1000 synthetic samples
- [ ] Save synthetic samples

**Implementation Approach**:
```python
from stg.tableSynthesizer import TableSynthesizer
import pandas as pd

# Load training data
train_df = pd.read_csv("train_data.csv")

# Initialize TabSyn with minimal config
config = {
    "epochs": 10,  # Adjust based on data size
    "batch_size": 32
}
tabsyn = TableSynthesizer('TabSyn', config=config)

# Train (accepts DataFrame directly)
print("Training TabSyn...")
tabsyn.fit(train_df)

# Save checkpoint
checkpoint = tabsyn.get_checkpoint()
torch.save(checkpoint, "checkpoints/tabsyn_checkpoint.pt")

# Generate 1000 samples
print("Generating 1000 synthetic samples...")
synthetic_samples = tabsyn.sample(n=1000, return_dataframe=True)

# Save synthetic data
synthetic_samples.to_csv("outputs/tabsyn_synthetic_1000.csv", index=False)
```

**Expected Output**:
- `checkpoints/tabsyn_checkpoint.pt`
- `outputs/tabsyn_synthetic_1000.csv`

---

### Step 3: Model Training - CTGAN
**Goal**: Train CTGAN synthesizer on training data

**Characteristics**:
- Input: DataFrame or DataLoader
- Approach: Conditional GAN with generator/discriminator
- Training: Direct PyTorch training
- Output: Tensor (convert to DataFrame)

**Tasks**:
- [ ] Initialize CTGAN synthesizer
- [ ] Train on `train_df`
- [ ] Save model checkpoint
- [ ] Generate 1000 synthetic samples
- [ ] Save synthetic samples

**Implementation Approach**:
```python
from stg.tableSynthesizer import TableSynthesizer
import pandas as pd
import torch

# Load training data
train_df = pd.read_csv("train_data.csv")

# Initialize CTGAN with minimal config
config = {
    "epochs": 10,
    "batch_size": 32,
    "embedding_dim": 128,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256)
}
ctgan = TableSynthesizer('CTGAN', config=config)

# Train (accepts DataFrame, automatically encodes)
print("Training CTGAN...")
ctgan.fit(train_df)

# Save checkpoint
checkpoint = ctgan.get_checkpoint()
torch.save(checkpoint, "checkpoints/ctgan_checkpoint.pt")

# Generate 1000 samples
print("Generating 1000 synthetic samples...")
synthetic_samples = ctgan.sample(n=1000, return_dataframe=True)

# Save synthetic data
synthetic_samples.to_csv("outputs/ctgan_synthetic_1000.csv", index=False)
```

**Expected Output**:
- `checkpoints/ctgan_checkpoint.pt`
- `outputs/ctgan_synthetic_1000.csv`

---

### Step 4: Model Training - GReat
**Goal**: Train GReat (Transformer-based) synthesizer on training data

**Characteristics**:
- Input: DataFrame only
- Approach: Synthcity-based transformer model
- Dependencies: Requires `synthcity` package
- Output: DataFrame (native)

**Tasks**:
- [ ] Verify `synthcity` package is installed
- [ ] Initialize GReat synthesizer
- [ ] Train on `train_df`
- [ ] Save model checkpoint
- [ ] Generate 1000 synthetic samples
- [ ] Save synthetic samples

**Implementation Approach**:
```python
from stg.tableSynthesizer import TableSynthesizer
import pandas as pd
import torch

# Load training data
train_df = pd.read_csv("train_data.csv")

# Initialize GReat with minimal config
config = {
    "epochs": 10,
    "batch_size": 32,
    # GReat-specific parameters (if needed)
}
great = TableSynthesizer('GReat', config=config)

# Train (accepts DataFrame directly)
print("Training GReat...")
great.fit(train_df)

# Save checkpoint
checkpoint = great.get_checkpoint()
torch.save(checkpoint, "checkpoints/great_checkpoint.pt")

# Generate 1000 samples
print("Generating 1000 synthetic samples...")
synthetic_samples = great.sample(n=1000, return_dataframe=True)

# Save synthetic data
synthetic_samples.to_csv("outputs/great_synthetic_1000.csv", index=False)
```

**Expected Output**:
- `checkpoints/great_checkpoint.pt`
- `outputs/great_synthetic_1000.csv`

**Note**: If `synthcity` is not installed, run:
```bash
pip install synthcity
```

---

### Step 5: Model Training - PATE-CTGAN
**Goal**: Train PATE-CTGAN (privacy-preserving) synthesizer on training data

**Characteristics**:
- Input: DataFrame or DataLoader
- Approach: CTGAN with PATE framework for differential privacy
- Training: Multi-student architecture with privacy budget
- Output: Tensor (convert to DataFrame)

**Tasks**:
- [ ] Initialize PATE-CTGAN synthesizer
- [ ] Train on `train_df`
- [ ] Save model checkpoint
- [ ] Generate 1000 synthetic samples
- [ ] Save synthetic samples

**Implementation Approach**:
```python
from stg.tableSynthesizer import TableSynthesizer
import pandas as pd
import torch

# Load training data
train_df = pd.read_csv("train_data.csv")

# Initialize PATE-CTGAN with minimal config
config = {
    "epochs": 10,
    "batch_size": 32,
    "num_teachers": 5,  # PATE-specific: number of teacher models
    "epsilon": 1.0,     # PATE-specific: privacy budget
    "delta": 1e-5       # PATE-specific: privacy parameter
}
pate_ctgan = TableSynthesizer('PATE-CTGAN', config=config)

# Train (accepts DataFrame, automatically encodes)
print("Training PATE-CTGAN...")
pate_ctgan.fit(train_df)

# Save checkpoint
checkpoint = pate_ctgan.get_checkpoint()
torch.save(checkpoint, "checkpoints/pate_ctgan_checkpoint.pt")

# Generate 1000 samples
print("Generating 1000 synthetic samples...")
synthetic_samples = pate_ctgan.sample(n=1000, return_dataframe=True)

# Save synthetic data
synthetic_samples.to_csv("outputs/pate_ctgan_synthetic_1000.csv", index=False)
```

**Expected Output**:
- `checkpoints/pate_ctgan_checkpoint.pt`
- `outputs/pate_ctgan_synthetic_1000.csv`

---

### Step 6: Evaluation (Optional)
**Goal**: Compare synthetic data quality across models

**Tasks**:
- [ ] Load all synthetic datasets
- [ ] Compute basic statistics (mean, std, distributions)
- [ ] Visual comparison (distributions, correlations)
- [ ] Quality metrics (if available)

**Implementation Approach**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all synthetic datasets
tabsyn_synth = pd.read_csv("outputs/tabsyn_synthetic_1000.csv")
ctgan_synth = pd.read_csv("outputs/ctgan_synthetic_1000.csv")
great_synth = pd.read_csv("outputs/great_synthetic_1000.csv")
pate_ctgan_synth = pd.read_csv("outputs/pate_ctgan_synthetic_1000.csv")

# Load original train/test data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Compare distributions
for col in train_df.columns:
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    train_df[col].hist(bins=30, alpha=0.7, label='Train')
    plt.title('Original (Train)')

    plt.subplot(2, 3, 2)
    tabsyn_synth[col].hist(bins=30, alpha=0.7, label='TabSyn')
    plt.title('TabSyn')

    plt.subplot(2, 3, 3)
    ctgan_synth[col].hist(bins=30, alpha=0.7, label='CTGAN')
    plt.title('CTGAN')

    plt.subplot(2, 3, 4)
    great_synth[col].hist(bins=30, alpha=0.7, label='GReat')
    plt.title('GReat')

    plt.subplot(2, 3, 5)
    pate_ctgan_synth[col].hist(bins=30, alpha=0.7, label='PATE-CTGAN')
    plt.title('PATE-CTGAN')

    plt.tight_layout()
    plt.savefig(f"outputs/comparison_{col}.png")
    plt.close()

print("Evaluation complete. Check outputs/ for comparison plots.")
```

**Expected Output**:
- Distribution comparison plots for each column
- Statistical summary reports

---

## Best Practices (No Hyperparameter Tuning)

### Default Configurations by Model

**TabSyn**:
```python
config = {
    "epochs": 10,
    "batch_size": 32
}
```
- **Rationale**: TabSyn uses subprocess-based VAE+Diffusion, minimal config works well for most datasets

**CTGAN**:
```python
config = {
    "epochs": 10,
    "batch_size": 32,
    "embedding_dim": 128,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256)
}
```
- **Rationale**: Standard CTGAN architecture from original paper

**GReat**:
```python
config = {
    "epochs": 10,
    "batch_size": 32
}
```
- **Rationale**: GReat handles most complexity internally via Synthcity

**PATE-CTGAN**:
```python
config = {
    "epochs": 10,
    "batch_size": 32,
    "num_teachers": 5,
    "epsilon": 1.0,
    "delta": 1e-5
}
```
- **Rationale**: Default privacy budget settings for moderate privacy-utility tradeoff

### Device Selection
```python
import torch

# Auto-detect best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Apply to synthesizer
synthesizer.set_device(device)
```

### Reproducibility
```python
import random
import numpy as np
import torch

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Apply to synthesizer
synthesizer.set_seed(seed)
```

---

## Complete End-to-End Script

Here's a consolidated script that trains all four models:

```python
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from stg.tableSynthesizer import TableSynthesizer
import os

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================
# Step 1: Data Preparation
# ============================================================
print("=== Step 1: Data Preparation ===")
data_path = "/home/ohsono/dataset/input_data/your_file.csv"
df = pd.read_csv(data_path)

# Split 80/20
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

# ============================================================
# Step 2: Train TabSyn
# ============================================================
print("\n=== Step 2: Training TabSyn ===")
config_tabsyn = {"epochs": 10, "batch_size": 32}
tabsyn = TableSynthesizer('TabSyn', config=config_tabsyn)
tabsyn.set_seed(42)
tabsyn.fit(train_df)

checkpoint = tabsyn.get_checkpoint()
torch.save(checkpoint, "checkpoints/tabsyn_checkpoint.pt")

synthetic_samples = tabsyn.sample(n=1000, return_dataframe=True)
synthetic_samples.to_csv("outputs/tabsyn_synthetic_1000.csv", index=False)
print("TabSyn complete: outputs/tabsyn_synthetic_1000.csv")

# ============================================================
# Step 3: Train CTGAN
# ============================================================
print("\n=== Step 3: Training CTGAN ===")
config_ctgan = {
    "epochs": 10,
    "batch_size": 32,
    "embedding_dim": 128,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256)
}
ctgan = TableSynthesizer('CTGAN', config=config_ctgan)
ctgan.set_seed(42)
ctgan.fit(train_df)

checkpoint = ctgan.get_checkpoint()
torch.save(checkpoint, "checkpoints/ctgan_checkpoint.pt")

synthetic_samples = ctgan.sample(n=1000, return_dataframe=True)
synthetic_samples.to_csv("outputs/ctgan_synthetic_1000.csv", index=False)
print("CTGAN complete: outputs/ctgan_synthetic_1000.csv")

# ============================================================
# Step 4: Train GReat
# ============================================================
print("\n=== Step 4: Training GReat ===")
config_great = {"epochs": 10, "batch_size": 32}
great = TableSynthesizer('GReat', config=config_great)
great.set_seed(42)
great.fit(train_df)

checkpoint = great.get_checkpoint()
torch.save(checkpoint, "checkpoints/great_checkpoint.pt")

synthetic_samples = great.sample(n=1000, return_dataframe=True)
synthetic_samples.to_csv("outputs/great_synthetic_1000.csv", index=False)
print("GReat complete: outputs/great_synthetic_1000.csv")

# ============================================================
# Step 5: Train PATE-CTGAN
# ============================================================
print("\n=== Step 5: Training PATE-CTGAN ===")
config_pate = {
    "epochs": 10,
    "batch_size": 32,
    "num_teachers": 5,
    "epsilon": 1.0,
    "delta": 1e-5
}
pate_ctgan = TableSynthesizer('PATE-CTGAN', config=config_pate)
pate_ctgan.set_seed(42)
pate_ctgan.fit(train_df)

checkpoint = pate_ctgan.get_checkpoint()
torch.save(checkpoint, "checkpoints/pate_ctgan_checkpoint.pt")

synthetic_samples = pate_ctgan.sample(n=1000, return_dataframe=True)
synthetic_samples.to_csv("outputs/pate_ctgan_synthetic_1000.csv", index=False)
print("PATE-CTGAN complete: outputs/pate_ctgan_synthetic_1000.csv")

print("\n=== All models trained and samples generated ===")
print("Checkpoints: checkpoints/")
print("Synthetic samples: outputs/")
```

---

## Expected File Structure

```
.
├── train_data.csv                          # Training set (80%)
├── test_data.csv                           # Test set (20%)
├── checkpoints/
│   ├── tabsyn_checkpoint.pt                # TabSyn model
│   ├── ctgan_checkpoint.pt                 # CTGAN model
│   ├── great_checkpoint.pt                 # GReat model
│   └── pate_ctgan_checkpoint.pt            # PATE-CTGAN model
├── outputs/
│   ├── tabsyn_synthetic_1000.csv           # TabSyn synthetic data
│   ├── ctgan_synthetic_1000.csv            # CTGAN synthetic data
│   ├── great_synthetic_1000.csv            # GReat synthetic data
│   └── pate_ctgan_synthetic_1000.csv       # PATE-CTGAN synthetic data
└── implementation_plan.md                  # This file
```

---

## Troubleshooting

### Issue 1: Missing Dependencies
**Symptom**: `ModuleNotFoundError` for `synthcity` or other packages

**Solution**:
```bash
pip install synthcity
pip install -e .
```

### Issue 2: CUDA Out of Memory
**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size or use CPU
```python
config = {"epochs": 10, "batch_size": 16}  # Reduced batch size
synthesizer.set_device("cpu")  # Use CPU instead
```

### Issue 3: TabSyn Subprocess Timeout
**Symptom**: TabSyn training hangs or times out

**Solution**: Check TabSyn subprocess logs and increase timeout if needed
```python
# TabSyn handles subprocess internally, check logs in temporary directory
# Or reduce epochs for faster training
config = {"epochs": 5, "batch_size": 32}
```

### Issue 4: Data Type Compatibility
**Symptom**: Encoding errors or type mismatches

**Solution**: Ensure DataFrame has proper dtypes before training
```python
# Convert categorical columns to 'category' dtype
for col in categorical_columns:
    train_df[col] = train_df[col].astype('category')

# Convert numerical columns to float
for col in numerical_columns:
    train_df[col] = train_df[col].astype('float32')
```

---

## Next Steps

1. **Run Data Preparation**: Load and split your dataset
2. **Train Models Sequentially**: Start with CTGAN (fastest), then TabSyn, GReat, PATE-CTGAN
3. **Verify Outputs**: Check that all models generate 1000 samples
4. **Optional Evaluation**: Compare synthetic data quality if needed
5. **Save Artifacts**: Keep checkpoints for future use or re-generation

---

## Notes

- **No Hyperparameter Tuning**: Using default/recommended configurations for all models
- **Reproducibility**: Set seed=42 for consistent results
- **Device**: Auto-detect CUDA/MPS/CPU (can override if needed)
- **Training Time**: Varies by model and dataset size (CTGAN fastest, TabSyn slowest due to subprocess)
- **Memory Usage**: PATE-CTGAN requires most memory due to multi-teacher architecture

---

**Created**: 2026-02-04
**Purpose**: Multi-model synthetic data generation without hyperparameter tuning
**Status**: Ready for implementation
