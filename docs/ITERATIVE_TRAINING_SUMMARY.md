# Iterative Training Feature - Quick Summary

**Date**: 2026-02-05
**Feature**: Batch processing of multiple datasets

---

## ✅ What Changed

Added **iterative training mode** to automatically train models on multiple datasets in a folder.

### Before (Single Dataset):
```bash
python train_all_compatible_models.py --dataset insurance --models CTGAN
```
- Train on ONE dataset
- Must specify dataset name
- Output in single directory

### After (Multiple Datasets):
```bash
python train_all_compatible_models.py \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN
```
- Train on ALL datasets in folder
- Auto-discovers all files
- Output organized by dataset

---

## 🚀 Quick Start

### Train on All CSV Files
```bash
python train_all_compatible_models.py \
    --data_folder ./datasets \
    --file_type csv \
    --iterate_datasets \
    --models CTGAN TVAE
```

### Train on All Parquet Files
```bash
python train_all_compatible_models.py \
    --data_folder ./data \
    --file_type parquet \
    --iterate_datasets \
    --group gpu
```

### Recursive Search
```bash
python train_all_compatible_models.py \
    --data_folder ./data \
    --iterate_datasets \
    --recursive \
    --models TVAE
```

---

## 📁 Output Structure

```
outputs/
├── dataset1/
│   ├── synthetic_ctgan_1000.csv
│   ├── synthetic_tvae_1000.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   └── training_summary_dataset1.csv
├── dataset2/
│   ├── synthetic_ctgan_1000.csv
│   └── ...
├── dataset3/
│   └── ...
└── combined_training_summary.csv  ← NEW!
```

---

## 🆕 New Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--iterate_datasets` | Enable iterative mode | (flag, no value) |
| `--file_type` | File type to search | `--file_type csv` or `--file_type parquet` |
| `--recursive` | Search subdirectories | (flag, no value) |

---

## 📊 Features

### Automatic Discovery
- ✅ Discovers all CSV or Parquet files in folder
- ✅ Supports recursive subdirectory search
- ✅ Automatically uses filename as dataset name

### Organized Output
- ✅ Each dataset gets own subdirectory
- ✅ Combined summary across all datasets
- ✅ Per-dataset training summaries

### Error Handling
- ✅ If one dataset fails, continues with others
- ✅ Clear error messages per dataset
- ✅ Overall success/failure summary

### Performance
- ✅ Independent batch size optimization per dataset
- ✅ Memory freed after each dataset
- ✅ Progress tracking across all datasets

---

## 🎯 Use Cases

1. **Batch Processing**: Process entire data warehouse
2. **Time-Series Data**: Train on Q1, Q2, Q3, Q4 separately
3. **Multi-Tenant**: Generate synthetic data per customer
4. **A/B Testing**: Compare quality across datasets
5. **Departmental Data**: Process HR, Sales, Marketing data

---

## 📝 Implementation Details

### Files Modified
- **train_all_compatible_models.py**: Added iterative mode logic

### Functions Added
- `discover_datasets()`: Find all datasets in folder
- `train_on_dataset()`: Train models on single dataset (refactored)

### Changes to main()
- Added dataset discovery logic
- Added iteration loop over datasets
- Added combined summary generation
- Maintained backward compatibility

---

## ✅ Validation

**Test 1: Multiple CSV Files**
```bash
python train_all_compatible_models.py \
    --data_folder ./test_datasets \
    --file_type csv \
    --iterate_datasets \
    --models TVAE \
    --epochs 1 \
    --samples 5

Result: ✅ Processed 2 datasets successfully
```

**Test 2: Parquet Files**
```bash
python train_all_compatible_models.py \
    --data_folder ./test_datasets \
    --file_type parquet \
    --iterate_datasets \
    --models TVAE \
    --epochs 1 \
    --samples 3

Result: ✅ Processed 1 parquet dataset successfully
```

**Test 3: Recursive Search**
```bash
python train_all_compatible_models.py \
    --data_folder ./test_datasets \
    --iterate_datasets \
    --recursive \
    --models TVAE \
    --epochs 1 \
    --samples 3

Result: ✅ Found and processed datasets in subdirectories
```

**Test 4: Single Dataset Mode (Backward Compatibility)**
```bash
python train_all_compatible_models.py \
    --dataset insurance \
    --data_folder ./data \
    --models TVAE

Result: ✅ Original single-dataset mode still works
```

---

## 🆚 Comparison: Single vs Iterative

| Feature | Single Mode | Iterative Mode |
|---------|------------|----------------|
| **Command** | `--dataset NAME` | `--iterate_datasets` |
| **Datasets** | 1 | Multiple (auto-discovered) |
| **File Discovery** | Manual | Automatic |
| **Output** | Single directory | Multiple directories |
| **Summary** | One CSV | Combined + individual CSVs |
| **Error Handling** | Stops | Continues with others |
| **Progress** | Model-level | Dataset + Model-level |

---

## 📚 Documentation

**Full Documentation**: See `ITERATIVE_TRAINING_GUIDE.md`

**Related Docs**:
- **ENHANCED_TRAINING_GUIDE.md**: Training guide
- **CONFIG_EXTERNALIZATION_SUMMARY.md**: Config guide
- **BATCH_SIZE_OPTIMIZATION_SUMMARY.md**: Batch optimization

---

## 🎓 Best Practices

1. **Test First**: Use `--epochs 1 --samples 10` for quick validation
2. **Start Small**: Process subset of datasets before full run
3. **Monitor Space**: Check disk space before large batch
4. **Use Fast Model**: TVAE for initial testing
5. **Analyze Summary**: Use combined_training_summary.csv for insights

---

## Example Output

```
============================================================
Iterative Dataset Mode
============================================================
Searching for csv files in: ./datasets

Found 3 dataset(s):
  1. insurance (datasets/insurance.csv)
  2. credit (datasets/credit.csv)
  3. loans (datasets/loans.csv)

############################################################
# Dataset 1/3
############################################################

Processing Dataset: insurance
...
✅ CTGAN completed in 120.5s

############################################################
# Dataset 2/3
############################################################

Processing Dataset: credit
...
✅ CTGAN completed in 180.3s

############################################################
# Dataset 3/3
############################################################

Processing Dataset: loans
...
✅ CTGAN completed in 150.2s

============================================================
OVERALL SUMMARY - ALL DATASETS
============================================================

✅ insurance: 1/1 successful, Time: 120.5s
✅ credit: 1/1 successful, Time: 180.3s
✅ loans: 1/1 successful, Time: 150.2s

Total Datasets: 3
Total Models Trained: 3 successful, 0 failed
Total Time: 451.0s (7.5 min, 0.13 hours)
============================================================
```

---

**Status**: Production-ready ✅
**Backward Compatible**: Yes ✅
**Tested**: CSV, Parquet, Recursive ✅
