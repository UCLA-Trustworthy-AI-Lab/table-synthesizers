# TabSyn POC - Quick Start

## What Was Created

I've created a complete proof-of-concept script for training TabSyn and generating synthetic data.

## Files Created

1. **`train_tabsyn_poc.py`** - Main POC script
2. **`TABSYN_POC_GUIDE.md`** - Comprehensive usage guide
3. **`implementation_plan.md`** - Full multi-model implementation plan
4. **`POC_SUMMARY.md`** - Detailed summary
5. **`QUICK_START.md`** - This file

## Current Status

A test run is currently executing in the background:
- Dataset: Titanic (714 samples)
- Training epochs: 2
- Synthetic samples to generate: 100

## Dependencies Installed

To make TabSyn work, I installed:
- `icecream` - Debugging utility
- `category_encoders` - Categorical encoding
- `tomli` - TOML parser
- All dependencies from `src/stg/TabSyn/requirements.txt`

## Quick Commands

### Run the POC with default settings (insurance dataset)
```bash
python train_tabsyn_poc.py
```

### Run with custom parameters
```bash
python train_tabsyn_poc.py --dataset Titanic --epochs 10 --samples 1000
```

### Check current training progress
```bash
# View live output of current run
tail -f /tmp/claude/-home-ohsono-Projects-table-synthesizers/tasks/b008288.output

# Or if saved to a file
tail -f tabsyn_run_complete.log
```

### After completion, view results
```bash
# View synthetic data
cat outputs/tabsyn_Titanic_synthetic_100.csv | head

# View training/test splits
cat data_splits/Titanic_train.csv | head
cat data_splits/Titanic_test.csv | head
```

## Available Datasets

Your input directory (`/home/ohsono/dataset/input_data/`) contains:

| Dataset | Size | Best For |
|---------|------|----------|
| Titanic | 20K | Quick testing (default POC) |
| IndianLiverPatient | 28K | Quick testing |
| insurance | 55K | **Recommended for POC** |
| abalone | 192K | Medium testing |
| Obesity | 271K | Medium testing |
| faults | 301K | Production testing |
| Bean | 2.4M | Large dataset testing |
| HTRU2 | 1.8M | Large dataset testing |

## What the Script Does

1. **Loads data** from specified CSV/Parquet file
2. **Splits 80/20** into train/test sets
3. **Trains TabSyn model** (VAE + Diffusion)
4. **Saves checkpoint** for later use
5. **Generates synthetic samples**
6. **Compares statistics** between real and synthetic data

## Expected Outputs

```
table-synthesizers/
├── data_splits/
│   ├── Titanic_train.csv          # 80% of data
│   └── Titanic_test.csv           # 20% of data
├── checkpoints/
│   └── tabsyn_Titanic_checkpoint.pt  # Trained model
├── outputs/
│   └── tabsyn_Titanic_synthetic_100.csv  # 100 synthetic samples
└── train_tabsyn_poc.py            # The script
```

## Estimated Training Times

| Dataset Size | Epochs=2 | Epochs=10 |
|-------------|----------|-----------|
| ~700 samples (Titanic) | 5-10 min | 25-50 min |
| ~1,300 samples (insurance) | 10-15 min | 50-75 min |
| ~4,000 samples (abalone) | 15-25 min | 75-125 min |

*Times are for CPU. GPU would be 5-10x faster.*

## Next Steps After Current Run Completes

1. **Check the results**:
   ```bash
   ls -lh outputs/
   ls -lh checkpoints/
   ```

2. **Inspect synthetic data**:
   ```bash
   # View first 10 rows
   head -10 outputs/tabsyn_Titanic_synthetic_100.csv

   # Count rows
   wc -l outputs/tabsyn_Titanic_synthetic_100.csv
   ```

3. **Run with larger dataset**:
   ```bash
   python train_tabsyn_poc.py --dataset insurance --epochs 10 --samples 1000
   ```

4. **Use saved checkpoint** to generate more samples without retraining:
   ```python
   import torch
   from stg.tableSynthesizer import TableSynthesizer

   # Load checkpoint
   checkpoint = torch.load("checkpoints/tabsyn_Titanic_checkpoint.pt")

   # Create synthesizer and load
   config = {"epochs": 2, "dataset_name": "tabsyn_Titanic"}
   synth = TableSynthesizer('TabSyn', config=config)
   synth.load_checkpoint(checkpoint)

   # Generate 5000 new samples
   new_samples = synth.sample(n=5000, return_dataframe=True)
   new_samples.to_csv("outputs/additional_5000.csv", index=False)
   ```

## Troubleshooting

### Check if training is still running
```bash
ps aux | grep train_tabsyn_poc
```

### View errors in log
```bash
grep -i error tabsyn_run_complete.log
grep -i "failed" tabsyn_run_complete.log
```

### Kill background process if needed
```bash
pkill -f train_tabsyn_poc
```

### Start fresh
```bash
rm -rf outputs/ checkpoints/ data_splits/
python train_tabsyn_poc.py --dataset insurance --epochs 2 --samples 100
```

## Production Usage

For production-quality synthetic data, use more epochs:

```bash
# High quality (recommended)
python train_tabsyn_poc.py --dataset insurance --epochs 15 --samples 5000

# Very high quality (slow)
python train_tabsyn_poc.py --dataset insurance --epochs 30 --samples 10000
```

## Integration with Your Workflow

This script demonstrates the core workflow. You can:

1. **Modify for batch processing**: Loop over multiple datasets
2. **Add custom preprocessing**: Edit `load_and_split_data()` function
3. **Custom evaluation**: Add metrics in `compare_statistics()` function
4. **Save to database**: Replace CSV save with database inserts
5. **Integrate with ML pipeline**: Use synthetic data for model training

## References

- Full guide: `TABSYN_POC_GUIDE.md`
- Implementation plan: `implementation_plan.md`
- Detailed summary: `POC_SUMMARY.md`

---

**Created**: 2026-02-04
**Status**: Test run in progress (task ID: b008288)
**Next**: Wait for completion, then inspect outputs
