# TabSyn Proof of Concept - Quick Start Guide

## Overview

This guide walks you through running the TabSyn proof of concept script to train a synthetic data generator and create 1000 synthetic samples.

## What This Script Does

1. **Loads data** from `/home/ohsono/dataset/input_data/`
2. **Splits data** into 80% training, 20% test
3. **Trains TabSyn model** using VAE + Diffusion approach
4. **Generates 1000 synthetic samples**
5. **Compares statistics** between real and synthetic data
6. **Saves all outputs** (checkpoints, synthetic data, splits)

## Quick Start

### Basic Usage (Default Settings)

Run with insurance dataset (default):

```bash
python train_tabsyn_poc.py
```

This will:
- Use `insurance` dataset
- Train for 10 epochs
- Generate 1000 synthetic samples
- Use random seed 42 for reproducibility

### Custom Dataset

Try different datasets from your input folder:

```bash
# Use Titanic dataset
python train_tabsyn_poc.py --dataset Titanic

# Use abalone dataset
python train_tabsyn_poc.py --dataset abalone

# Use Bean dataset (larger)
python train_tabsyn_poc.py --dataset Bean
```

### Custom Parameters

```bash
# Quick test: 2 epochs, 100 samples
python train_tabsyn_poc.py --epochs 2 --samples 100

# Production run: 20 epochs, 5000 samples
python train_tabsyn_poc.py --epochs 20 --samples 5000

# Full customization
python train_tabsyn_poc.py \
    --dataset insurance \
    --epochs 15 \
    --samples 2000 \
    --seed 123
```

### All Available Options

```bash
python train_tabsyn_poc.py --help
```

**Options:**
- `--dataset`: Dataset name (default: insurance)
- `--epochs`: Training epochs (default: 10)
- `--samples`: Number of synthetic samples (default: 1000)
- `--input-dir`: Input data directory (default: /home/ohsono/dataset/input_data/)
- `--seed`: Random seed (default: 42)

## Available Datasets

Your input directory contains:

| Dataset | Size | Description |
|---------|------|-------------|
| abalone | 192K | Marine biology data |
| Bean | 2.4M | Bean classification |
| faults | 301K | Fault detection |
| HTRU2 | 1.8M | Pulsar star classification |
| IndianLiverPatient | 28K | Liver disease data |
| **insurance** | 55K | **Health insurance (default)** |
| News | 15M | News article data (large) |
| Obesity | 271K | Obesity classification |
| Shoppers | 1.1M | Online shopping behavior |
| Titanic | 20K | Titanic survival data |
| wilt | 307K | Plant disease data |

## Output Structure

After running the script, you'll have:

```
table-synthesizers/
├── data_splits/
│   ├── insurance_train.csv          # 80% training data
│   └── insurance_test.csv           # 20% test data
├── checkpoints/
│   └── tabsyn_insurance_checkpoint.pt  # Trained model
├── outputs/
│   └── tabsyn_insurance_synthetic_1000.csv  # Synthetic data
├── train_tabsyn_poc.py              # This script
└── TABSYN_POC_GUIDE.md             # This guide
```

## Expected Output

When you run the script, you'll see:

```
============================================================
TabSyn Proof of Concept
============================================================
  Dataset: insurance
  Training epochs: 10
  Synthetic samples: 1000
  Random seed: 42
============================================================

✓ Created directory: outputs/
✓ Created directory: checkpoints/
✓ Created directory: data_splits/

============================================================
STEP 1: Data Loading and Splitting
============================================================
✓ Loaded CSV: /home/ohsono/dataset/input_data/insurance.csv
  Dataset shape: (1338, 7)
  Columns: ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']

  Splitting data: 80% train, 20% test
✓ Training set: 1070 samples
✓ Test set: 268 samples
✓ Saved train split: data_splits/insurance_train.csv
✓ Saved test split: data_splits/insurance_test.csv

============================================================
STEP 2: Training TabSyn Model
============================================================
  Configuration:
    - Model: TabSyn (VAE + Diffusion)
    - Epochs: 10
    - Training samples: 1070
    - Random seed: 42

  Starting training (this may take a while)...
  Note: TabSyn trains VAE first, then diffusion model
------------------------------------------------------------
[TabSyn training output...]
------------------------------------------------------------
✓ Training complete!

============================================================
STEP 3: Saving Model Checkpoint
============================================================
✓ Checkpoint saved successfully (12.34 MB)

============================================================
STEP 4: Generating Synthetic Samples
============================================================
  Generating 1000 synthetic samples...
------------------------------------------------------------
✓ Generated 1000 synthetic samples
  Synthetic data shape: (1000, 7)
✓ Saved synthetic data: outputs/tabsyn_insurance_synthetic_1000.csv

============================================================
STEP 5: Data Quality Comparison
============================================================

  Comparing distributions (Training vs Synthetic):

  Numerical Columns:
  --------------------------------------------------------
  Column               Train Mean      Synth Mean
  --------------------------------------------------------
  age                  39.21           39.45
  bmi                  30.66           30.52
  children             1.09            1.11
  charges              13270.42        13156.78
  --------------------------------------------------------

  Categorical Columns:
  --------------------------------------------------------

  sex:
    Value           Train %     Synth %
    ---------------------------------------
    female          49.3        50.1
    male            50.7        49.9

  smoker:
    Value           Train %     Synth %
    ---------------------------------------
    no              79.5        80.2
    yes             20.5        19.8

  region:
    Value           Train %     Synth %
    ---------------------------------------
    northeast       24.1        23.8
    northwest       24.5        24.9
    southeast       27.4        27.1
    southwest       24.0        24.2
  --------------------------------------------------------

============================================================
SUMMARY
============================================================
✓ Training data: 1070 samples
✓ Test data: 268 samples
✓ Model checkpoint: checkpoints/tabsyn_insurance_checkpoint.pt
✓ Synthetic samples: 1000 rows
✓ Output: outputs/tabsyn_insurance_synthetic_1000.csv

============================================================
TabSyn Proof of Concept Complete!
============================================================
```

## Training Time Estimates

Approximate training times (varies by hardware):

| Dataset Size | CPU | GPU (CUDA) |
|-------------|-----|------------|
| ~1K samples (insurance, Titanic) | 15-30 min | 3-5 min |
| ~5K samples (Bean, faults) | 45-90 min | 10-15 min |
| ~10K+ samples (HTRU2, Shoppers) | 2-4 hours | 30-60 min |

**Note**: First run may take longer as TabSyn prepares dataset metadata.

## Understanding the Output

### 1. Data Splits
- **train_data.csv**: 80% of original data used for training
- **test_data.csv**: 20% held out for evaluation (not used in training)

### 2. Model Checkpoint
- **checkpoint.pt**: Trained TabSyn model weights
- Can be reloaded later to generate more samples without retraining

### 3. Synthetic Data
- **synthetic_1000.csv**: Generated synthetic samples
- Should have similar statistical properties to training data
- Can be used as privacy-preserving alternative to real data

### 4. Quality Metrics
The script compares:
- **Numerical columns**: Mean values (should be close)
- **Categorical columns**: Category distributions (should match percentages)

**Good quality indicators**:
- Means within 5-10% of original
- Category distributions within 2-5 percentage points
- No missing or impossible values

## Next Steps

After running the proof of concept:

1. **Evaluate Quality**: Inspect the synthetic data CSV and compare statistics
2. **Adjust Parameters**: Try different epoch counts if quality is poor
3. **Try Other Datasets**: Test on Titanic, abalone, or other datasets
4. **Scale Up**: Once satisfied, train with more epochs for production use

## Troubleshooting

### "ImportError: TabSyn dependencies are required"
**Solution**: Install package dependencies
```bash
pip install -e .
```

### "FileNotFoundError: Dataset 'X' not found"
**Solution**: Check available datasets
```bash
ls /home/ohsono/dataset/input_data/
```

### Training takes too long
**Solution**: Reduce epochs for testing
```bash
python train_tabsyn_poc.py --epochs 2 --samples 100
```

### Out of memory error
**Solution**: Try smaller dataset or use CPU
```bash
# Try smaller dataset
python train_tabsyn_poc.py --dataset Titanic

# Or reduce batch processing (modify config in script)
```

## Advanced Usage

### Load Checkpoint and Generate More Samples

To generate more samples without retraining:

```python
import torch
from stg.tableSynthesizer import TableSynthesizer

# Load checkpoint
checkpoint = torch.load("checkpoints/tabsyn_insurance_checkpoint.pt")

# Create new synthesizer and load state
config = {"epochs": 10, "dataset_name": "tabsyn_insurance"}
synthesizer = TableSynthesizer('TabSyn', config=config)
synthesizer.load_checkpoint(checkpoint)

# Generate new samples
new_samples = synthesizer.sample(n=5000, return_dataframe=True)
new_samples.to_csv("outputs/additional_samples_5000.csv", index=False)
```

## Questions?

For more information:
- See `implementation_plan.md` for multi-model comparison
- Check TabSyn source: `src/stg/TabSyn/tabsyn_synthesizer.py`
- Review integration tests: `tests/integration/test_tabsyn_integration.py`
