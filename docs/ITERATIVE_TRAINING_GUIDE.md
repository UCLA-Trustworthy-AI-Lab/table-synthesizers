# Iterative Dataset Training - Guide

**Date**: 2026-02-05
**Feature**: Batch processing of multiple datasets in a folder

---

## ✅ What Was Implemented

### Iterative Training Mode

Train models on **multiple datasets automatically** by providing a folder containing multiple data files. The script will:
1. Discover all datasets of specified file type (CSV or Parquet)
2. Train selected models on each dataset
3. Generate synthetic data for each dataset
4. Save results in organized subdirectories
5. Create combined summary across all datasets

---

## 🚀 Quick Start

### Single Dataset Mode (Original)
```bash
# Train on one specific dataset
python train_all_compatible_models.py \
    --dataset insurance \
    --data_folder ./data \
    --models CTGAN TVAE \
    --epochs 50 \
    --samples 1000
```

### Iterative Dataset Mode (NEW)
```bash
# Train on ALL datasets in folder
python train_all_compatible_models.py \
    --data_folder ./datasets \
    --file_type csv \
    --iterate_datasets \
    --models CTGAN TVAE \
    --epochs 50 \
    --samples 1000
```

---

## 📖 Usage Examples

### Example 1: Train on All CSV Files
```bash
# Folder structure:
# datasets/
#   ├── insurance.csv
#   ├── credit.csv
#   └── loans.csv

python train_all_compatible_models.py \
    --data_folder ./datasets \
    --file_type csv \
    --iterate_datasets \
    --models CTGAN \
    --epochs 50 \
    --samples 1000
```

**Output:**
```
Found 3 dataset(s):
  1. insurance (datasets/insurance.csv)
  2. credit (datasets/credit.csv)
  3. loans (datasets/loans.csv)

Training on each dataset...

outputs/
├── insurance/
│   ├── synthetic_ctgan_1000.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   └── training_summary_insurance.csv
├── credit/
│   ├── synthetic_ctgan_1000.csv
│   └── ...
├── loans/
│   ├── synthetic_ctgan_1000.csv
│   └── ...
└── combined_training_summary.csv
```

### Example 2: Train on All Parquet Files
```bash
python train_all_compatible_models.py \
    --data_folder ./data_warehouse \
    --file_type parquet \
    --iterate_datasets \
    --group gpu \
    --epochs 100
```

### Example 3: Recursive Search in Subdirectories
```bash
# Folder structure:
# datasets/
#   ├── 2023/
#   │   ├── q1_data.csv
#   │   └── q2_data.csv
#   └── 2024/
#       ├── q1_data.csv
#       └── q2_data.csv

python train_all_compatible_models.py \
    --data_folder ./datasets \
    --file_type csv \
    --iterate_datasets \
    --recursive \
    --models TVAE \
    --epochs 50
```

**Output:**
```
Found 4 dataset(s):
  1. q1_data (datasets/2023/q1_data.csv)
  2. q2_data (datasets/2023/q2_data.csv)
  3. q1_data (datasets/2024/q1_data.csv)
  4. q2_data (datasets/2024/q2_data.csv)
```

### Example 4: Quick Test on Multiple Datasets
```bash
# Fast validation - 1 epoch, 10 samples
python train_all_compatible_models.py \
    --data_folder ./test_data \
    --file_type csv \
    --iterate_datasets \
    --models TVAE \
    --epochs 1 \
    --samples 10
```

---

## 🔧 Command-Line Arguments

### New Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--iterate_datasets` | flag | False | Enable iterative mode (process all datasets in folder) |
| `--file_type` | str | csv | File type to search for: 'csv' or 'parquet' |
| `--recursive` | flag | False | Search subdirectories recursively |

### Existing Arguments (Still Work)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_folder` | str | (required) | Folder containing datasets |
| `--dataset` | str | None | Specific dataset (single mode only) |
| `--models` | list | None | Models to train (e.g., CTGAN TVAE) |
| `--group` | str | gpu | Model group: gpu, cpu, all |
| `--epochs` | int | Model default | Number of training epochs |
| `--samples` | int | 1000 | Synthetic samples to generate |
| `--output_dir` | str | ./outputs | Output directory |
| `--config_dir` | str | ./config | Config directory |
| `--test_size` | float | 0.2 | Train/test split ratio |

---

## 📁 Output Structure

### Single Dataset Mode
```
outputs/
├── {dataset_name}/
│   ├── synthetic_{model}_{samples}.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── training_summary_{dataset}.csv
│   └── model_configs_reference.txt
```

### Iterative Dataset Mode
```
outputs/
├── {dataset1}/
│   ├── synthetic_ctgan_1000.csv
│   ├── synthetic_tvae_1000.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── training_summary_{dataset1}.csv
│   └── model_configs_reference.txt
├── {dataset2}/
│   ├── synthetic_ctgan_1000.csv
│   ├── synthetic_tvae_1000.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── training_summary_{dataset2}.csv
│   └── model_configs_reference.txt
├── {dataset3}/
│   └── ...
└── combined_training_summary.csv  ← NEW: All datasets combined
```

### Combined Training Summary (CSV)
```csv
dataset,model,model_type,status,training_time_seconds,training_time_minutes,samples_generated,config
insurance,CTGAN,GPU,success,120.5,2.01,1000,"{'epochs': 50, ...}"
insurance,TVAE,GPU,success,15.2,0.25,1000,"{'epochs': 100, ...}"
credit,CTGAN,GPU,success,180.3,3.01,1000,"{'epochs': 50, ...}"
credit,TVAE,GPU,success,20.1,0.34,1000,"{'epochs': 100, ...}"
```

---

## 🎯 Use Cases

### 1. Batch Processing Multiple Datasets
```bash
# Process entire data warehouse
python train_all_compatible_models.py \
    --data_folder /data/warehouse \
    --file_type parquet \
    --iterate_datasets \
    --group gpu \
    --epochs 50
```

**Use Case**: Generate synthetic data for multiple tables in a database

### 2. Time-Series Data (Quarterly/Monthly)
```bash
# Process all quarterly datasets
python train_all_compatible_models.py \
    --data_folder ./financial_data/2024 \
    --file_type csv \
    --iterate_datasets \
    --models CTGAN \
    --epochs 50
```

**Use Case**: Train on Q1, Q2, Q3, Q4 data separately

### 3. A/B Testing Different Datasets
```bash
# Compare synthetic data quality across datasets
python train_all_compatible_models.py \
    --data_folder ./experiment_data \
    --iterate_datasets \
    --models CTGAN TVAE TabDDPM \
    --epochs 100
```

**Use Case**: Evaluate which dataset produces best synthetic data

### 4. Multi-Tenant SaaS Applications
```bash
# Generate synthetic data for each customer
python train_all_compatible_models.py \
    --data_folder ./customer_data \
    --iterate_datasets \
    --recursive \
    --models CTGAN \
    --epochs 50
```

**Use Case**: Each customer gets their own synthetic dataset

### 5. Departmental Data Processing
```bash
# Folder structure:
# company_data/
#   ├── sales/sales.csv
#   ├── marketing/campaigns.csv
#   ├── hr/employees.csv
#   └── finance/transactions.csv

python train_all_compatible_models.py \
    --data_folder ./company_data \
    --iterate_datasets \
    --recursive \
    --group all
```

**Use Case**: Process data from different departments

---

## ⚙️ Configuration

### File Type Selection

**CSV Files (default)**:
```bash
python train_all_compatible_models.py \
    --data_folder ./data \
    --file_type csv \
    --iterate_datasets
```

**Parquet Files**:
```bash
python train_all_compatible_models.py \
    --data_folder ./data \
    --file_type parquet \
    --iterate_datasets
```

### Search Scope

**Top-Level Only (default)**:
```bash
# Only searches data_folder/*.csv
python train_all_compatible_models.py \
    --data_folder ./data \
    --iterate_datasets
```

**Recursive Search**:
```bash
# Searches data_folder/**/*.csv (all subdirectories)
python train_all_compatible_models.py \
    --data_folder ./data \
    --iterate_datasets \
    --recursive
```

### Output Organization

**Default Output**:
```bash
# Creates outputs/{dataset1}, outputs/{dataset2}, etc.
python train_all_compatible_models.py \
    --data_folder ./data \
    --iterate_datasets
```

**Custom Output Directory**:
```bash
# Creates /results/{dataset1}, /results/{dataset2}, etc.
python train_all_compatible_models.py \
    --data_folder ./data \
    --iterate_datasets \
    --output_dir /results
```

---

## 🔍 Advanced Features

### Error Handling

If one dataset fails, the script continues with others:

```
Processing Dataset: insurance
✅ CTGAN completed in 120.5s

Processing Dataset: bad_data
❌ Error processing dataset bad_data: Invalid column types

Processing Dataset: loans
✅ CTGAN completed in 150.2s

OVERALL SUMMARY:
✅ insurance: 1/1 successful
❌ bad_data: Failed - Invalid column types
✅ loans: 1/1 successful
```

### Progress Tracking

Clear progress indicators for multiple datasets:

```
############################################################
# Dataset 1/3
############################################################
Processing Dataset: insurance
...

############################################################
# Dataset 2/3
############################################################
Processing Dataset: credit
...

############################################################
# Dataset 3/3
############################################################
Processing Dataset: loans
...
```

### Combined Summary Statistics

```
OVERALL SUMMARY - ALL DATASETS
============================================================

✅ insurance:
   Models: 2/2 successful
   Time: 135.7s (2.3 min)
   Output: outputs/insurance

✅ credit:
   Models: 2/2 successful
   Time: 200.3s (3.3 min)
   Output: outputs/credit

✅ loans:
   Models: 2/2 successful
   Time: 165.1s (2.8 min)
   Output: outputs/loans

============================================================
Total Datasets: 3
Total Models Trained: 6 successful, 0 failed
Total Time: 501.1s (8.4 min, 0.14 hours)
============================================================
```

---

## 📊 Performance Considerations

### Batch Size Optimization Per Dataset

Each dataset gets independent batch size optimization:

```
Processing Dataset: large_dataset (10K samples)
Batch size optimized: 200 → 128

Processing Dataset: small_dataset (500 samples)
Batch size optimized: 200 → 64
```

### Training Time Estimation

**Example: 3 datasets, 2 models each, 50 epochs**

| Dataset | Samples | Models | Time per Model | Total Time |
|---------|---------|--------|----------------|------------|
| insurance | 1,338 | CTGAN, TVAE | 10min, 30s | ~10.5 min |
| credit | 30,000 | CTGAN, TVAE | 45min, 2min | ~47 min |
| loans | 5,000 | CTGAN, TVAE | 20min, 1min | ~21 min |
| **Total** | - | - | - | **~78.5 min** |

**Sequential Processing**: Datasets processed one at a time
**Parallel Processing**: Not yet supported (future enhancement)

### Memory Management

**Per-Dataset Memory Usage**:
- Each dataset loads independently
- Memory freed after processing each dataset
- Safe for large dataset collections

**Disk Space Requirements**:
```
Disk usage = N_datasets × N_models × (
    synthetic_samples_size +
    train_data_size +
    test_data_size
)

Example (3 datasets, 2 models, 1K synthetic samples):
≈ 3 × 2 × (50KB + 40KB + 10KB) = 600KB total
```

---

## 🚨 Error Handling

### Missing Files

**Error:**
```
Error: No csv files found in ./data
```

**Solution:**
```bash
# Check folder contains files
ls ./data/*.csv

# Or try parquet
--file_type parquet
```

### Mixed File Types

**Scenario**: Folder has both CSV and Parquet files

**Solution**: Specify file type explicitly
```bash
# Only CSV
python train_all_compatible_models.py \
    --data_folder ./data \
    --file_type csv \
    --iterate_datasets

# Only Parquet
python train_all_compatible_models.py \
    --data_folder ./data \
    --file_type parquet \
    --iterate_datasets
```

### Dataset Name Conflicts

**Scenario**: Two datasets with same name in different subdirectories

```
data/
├── 2023/sales.csv
└── 2024/sales.csv
```

**Behavior**: Both will be processed, but output directories will conflict

**Solution**: Rename files or use non-recursive mode

---

## 🎓 Best Practices

### 1. Test Before Production

```bash
# Quick test with minimal parameters
python train_all_compatible_models.py \
    --data_folder ./data \
    --iterate_datasets \
    --models TVAE \
    --epochs 1 \
    --samples 10
```

### 2. Use Fast Model for Initial Testing

```bash
# TVAE is fastest - use for validation
python train_all_compatible_models.py \
    --data_folder ./data \
    --iterate_datasets \
    --models TVAE \
    --epochs 1
```

### 3. Monitor Disk Space

```bash
# Check available space before starting
df -h ./outputs

# Estimate required space
# N_datasets × N_models × average_file_size
```

### 4. Process Subsets for Large Collections

```bash
# Process 2024 data only
python train_all_compatible_models.py \
    --data_folder ./historical_data/2024 \
    --iterate_datasets

# Then process 2023 data
python train_all_compatible_models.py \
    --data_folder ./historical_data/2023 \
    --iterate_datasets
```

### 5. Use Combined Summary for Analysis

```python
import pandas as pd

# Load combined summary
summary = pd.read_csv('outputs/combined_training_summary.csv')

# Compare training times across datasets
summary.groupby('dataset')['training_time_minutes'].sum()

# Compare model performance
summary.groupby('model')['training_time_minutes'].mean()
```

---

## 📈 Comparison: Single vs Iterative Mode

| Feature | Single Dataset Mode | Iterative Dataset Mode |
|---------|-------------------|------------------------|
| **Setup** | Specify `--dataset` | Specify `--iterate_datasets` |
| **Input** | One dataset | Multiple datasets |
| **Output** | One directory | Multiple directories |
| **Summary** | Single CSV | Combined + individual CSVs |
| **Error Handling** | Stops on error | Continues with other datasets |
| **Use Case** | One-off training | Batch processing |
| **Command Complexity** | Simple | Slightly more complex |

---

## 🔮 Future Enhancements (Roadmap)

### Planned Features:
1. **Parallel Processing**: Process multiple datasets concurrently
2. **Dataset Filtering**: Include/exclude patterns (e.g., `--include "*2024*"`)
3. **Resumable Training**: Skip already-processed datasets
4. **Progress Bar**: Visual progress indicator with ETA
5. **Email Notifications**: Alert when batch complete
6. **Cloud Storage**: Direct S3/GCS dataset discovery

---

## 📚 Related Documentation

- **ENHANCED_TRAINING_GUIDE.md**: Comprehensive training guide
- **CONFIG_EXTERNALIZATION_SUMMARY.md**: Model configuration guide
- **BATCH_SIZE_OPTIMIZATION_SUMMARY.md**: Batch size optimization details

---

## ✅ Summary

**Key Benefits:**
- ✅ Process multiple datasets automatically
- ✅ Organized output per dataset
- ✅ Combined summary across all datasets
- ✅ Error isolation (one failure doesn't stop others)
- ✅ Support for CSV and Parquet files
- ✅ Recursive subdirectory search
- ✅ Same model configs across all datasets

**When to Use:**
- Batch processing data warehouse tables
- Time-series data (quarterly, monthly)
- Multi-tenant applications
- A/B testing datasets
- Departmental data processing

**Status**: Production-ready ✅
