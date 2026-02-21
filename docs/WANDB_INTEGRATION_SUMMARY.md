# WandB Integration - Quick Summary

**Date**: 2026-02-05
**Feature**: Weights & Biases metrics tracking and visualization

---

## ✅ What Was Added

### WandB Integration to Training Script

**Comprehensive metrics logging** for:
- Training time and performance
- Batch size optimization
- Dataset statistics
- Device information
- Model configurations
- Success/failure tracking
- Summary statistics

---

## 🚀 Quick Start

### 1. Install WandB
```bash
pip install wandb
```

### 2. Setup .env File
```bash
# Copy example
cp .env.example .env

# Edit with your API key
nano .env

# Add:
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=table-synthesizers
```

### 3. Train with WandB
```bash
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN TVAE \
    --use_wandb
```

---

## 📊 Metrics Logged

### Per-Model Metrics (under `{model}/` namespace)
- `dataset_samples` - Training sample count
- `dataset_features` - Feature/column count
- `batch_size_original` - Config batch size
- `batch_size_optimized` - Optimized batch size
- `training_time_seconds` - Training duration
- `training_time_minutes` - Training duration (min)
- `generation_time_seconds` - Synthetic data generation time
- `synthetic_samples_generated` - Samples created
- `samples_per_second` - Training throughput
- `device_type` - Device used (CUDA/MPS/CPU)
- `status` - Success (1) or failure (0)
- `config_{param}` - Model hyperparameters

### Summary Metrics (under `summary/` namespace)
- `total_models` - Models trained
- `successful_models` - Successful trainings
- `failed_models` - Failed trainings
- `total_training_time_seconds` - Total time
- `average_time_per_model` - Avg time per model
- `summary_table` - Results table

---

## 🆕 New Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--use_wandb` | Enable WandB logging | (flag) |
| `--wandb_project` | Project name | `--wandb_project my-experiments` |
| `--wandb_entity` | Team/entity name | `--wandb_entity my-team` |

---

## 📁 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | API key | (required) |
| `WANDB_PROJECT` | Project name | `table-synthesizers` |
| `WANDB_ENTITY` | Entity/team | None |
| `WANDB_MODE` | Mode | `online` |

---

## 🎯 Usage Examples

### Example 1: Basic WandB Logging
```bash
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN \
    --use_wandb
```

### Example 2: Custom Project
```bash
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN TVAE \
    --use_wandb \
    --wandb_project my-experiments
```

### Example 3: Iterative Mode with WandB
```bash
python train_all_compatible_models.py \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN \
    --use_wandb
```

**Creates one run per dataset** for easy comparison.

### Example 4: Offline Mode
```bash
# Set offline mode
export WANDB_MODE=offline

# Train
python train_all_compatible_models.py \
    --dataset insurance \
    --use_wandb

# Sync later
wandb sync ./wandb/run-XXXXX
```

---

## 📈 WandB Dashboard

### What You'll See

**Run Overview:**
- Run name: `{dataset}_{models}`
- Tags: dataset name, model names, `table-synthesizers`
- Config: dataset info, model list, parameters

**Metrics:**
- Training time per model
- Batch size optimization
- Success/failure status
- Generation time
- Samples per second

**Summary Table:**
| Model | Type | Status | Training Time | Generation Time |
|-------|------|--------|---------------|-----------------|
| CTGAN | GPU | success | 120.5s | 2.3s |
| TVAE | GPU | success | 15.2s | 0.8s |

---

## 🎓 Key Features

### Hierarchical Metrics
```
CTGAN/training_time_seconds
CTGAN/batch_size_optimized
TVAE/training_time_seconds
TVAE/batch_size_optimized
summary/total_models
summary/successful_models
```

Clear organization for easy filtering and comparison.

### Auto-Generated Run Names
```
insurance_CTGAN-TVAE
credit_CTGAN
loans_TVAE-TabDDPM
```

Descriptive names for quick identification.

### Automatic Tags
```
['insurance', 'table-synthesizers', 'CTGAN', 'TVAE']
```

Easy filtering and grouping.

---

## 🔍 Use Cases

### 1. Compare Models
Track which model trains fastest:
```bash
python train_all_compatible_models.py \
    --dataset insurance \
    --group gpu \
    --use_wandb
```

### 2. Monitor Optimization
See batch size optimization impact:
```bash
python train_all_compatible_models.py \
    --dataset large_data \
    --models CTGAN \
    --use_wandb
```

### 3. Multi-Dataset Experiments
Compare across datasets:
```bash
python train_all_compatible_models.py \
    --data_folder ./datasets \
    --iterate_datasets \
    --use_wandb
```

### 4. Production Monitoring
Track production runs:
```bash
python train_all_compatible_models.py \
    --dataset prod_data \
    --models CTGAN \
    --epochs 100 \
    --use_wandb \
    --wandb_project production
```

---

## ⚠️ Important Notes

### Optional Dependency
WandB is **optional** - script works without it:
```python
# If wandb not installed
Warning: wandb not installed. Install with: pip install wandb

# Script continues without WandB
```

### Graceful Degradation
If WandB fails to initialize:
```
⚠️  WandB initialization failed: ...
# Training continues normally
```

### Multiple Runs in Iterative Mode
Each dataset gets its own WandB run:
```
3 datasets → 3 WandB runs
```

This is **correct** for comparing across datasets.

---

## 🔧 Implementation Details

### Files Modified
- **train_all_compatible_models.py**

### Functions Updated
- `train_and_generate()` - Added metric logging
- `train_on_dataset()` - Added run initialization/finalization
- `main()` - Added command-line arguments

### New Files
- **.env.example** - Configuration template
- **WANDB_INTEGRATION_GUIDE.md** - Full documentation
- **WANDB_INTEGRATION_SUMMARY.md** - This file

---

## 🆚 Before vs After

### Before (No WandB)
```bash
python train_all_compatible_models.py --dataset insurance --models CTGAN

# Only console output and CSV files
# No real-time monitoring
# No comparison across runs
```

### After (With WandB)
```bash
python train_all_compatible_models.py --dataset insurance --models CTGAN --use_wandb

# Console output + CSV files + WandB dashboard
# Real-time metric tracking
# Cross-run comparisons
# Team collaboration
# Historical tracking
```

---

## 📚 Full Documentation

See **WANDB_INTEGRATION_GUIDE.md** for:
- Complete metrics list
- Dashboard setup
- Advanced configuration
- Troubleshooting
- Best practices

---

## ✅ Summary

**Added:**
- ✅ 20+ metrics per model
- ✅ Summary statistics
- ✅ Real-time tracking
- ✅ Offline mode support
- ✅ Environment variable configuration
- ✅ Graceful error handling
- ✅ Optional dependency

**Benefits:**
- ✅ Compare model performance
- ✅ Track batch size optimization
- ✅ Monitor production training
- ✅ Collaborate with team
- ✅ Historical analysis

**Status**: Production-ready ✅
**Backward Compatible**: Yes (optional feature) ✅
