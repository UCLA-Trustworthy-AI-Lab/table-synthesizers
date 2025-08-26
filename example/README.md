# Table Synthesizers - Examples & Benchmarks

This directory contains examples and benchmarking tools for the table synthesizers library with realistic datasets and production-ready configurations.

## Scripts

### `benchmark_synthesizers.py` - Production Synthesizer Benchmark
Comprehensive evaluation tool with realistic hyperparameters designed for production use on substantial datasets.

**Usage:**
```bash
# Navigate to example directory first
cd /path/to/table-synthesizers/example

# Then run benchmarks
python benchmark_synthesizers.py [--dataset FILE] [--n_samples N] [--timeout T]
```

**Options:**
- `--dataset FILE`: Dataset from sandbox/ folder (default: conversions_all_8-1-25.csv)
- `--n_samples N`: Synthetic samples to generate (default: same as input, max 10k)
- `--timeout T`: Timeout per synthesizer in seconds (default: 1800)
- `--output_dir DIR`: Output directory (default: benchmark_results)
- `--all_synthesizers`: Run all synthesizers regardless of dataset size
- `--verbose`: Show detailed configuration

**Recommended Commands:**

```bash
# 1. QUICK START: Small dataset, selected synthesizers (~15-30 minutes)
python benchmark_synthesizers.py

# 2. FULL BENCHMARK: Large dataset with all synthesizers (~2-4 hours)
python benchmark_synthesizers.py --dataset dsp_impressions_7-29-25.csv --n_samples 5000 --timeout 3600 --all_synthesizers

# 3. HIGH-DIMENSIONAL TEST: Complex dataset (182 columns, ~1-3 hours)
python benchmark_synthesizers.py --dataset amazon_attributed_events_by_traffic_time_7-29-25.csv --timeout 3600

# 4. COMPREHENSIVE EVALUATION: All datasets, all synthesizers (automated script)
./run_comprehensive_benchmark.sh

# Or with custom timeout (in seconds)
./run_comprehensive_benchmark.sh 7200  # 2 hours per synthesizer
```

**Key Features:**
- **Smart Synthesizer Selection**: Automatically chooses optimal synthesizers based on dataset characteristics
- **Production Hyperparameters**: 
  - CTGAN: 300 epochs, 2e-4 LR, (256,256) layers
  - TabDDPM: 1000 timesteps, 1000 epochs
  - GREAT: 1000 epochs, transformer optimizations
  - TabSyn: 100 epochs for VAE+Diffusion
- **Performance Monitoring**: Tracks fit time, generation time, peak memory usage
- **Quality Assessment**: Column preservation, data type matching, shape validation
- **Comprehensive Reporting**: JSON reports with performance rankings and detailed metrics

### `run_comprehensive_benchmark.sh` - Automated Full Evaluation
Automated shell script that runs all synthesizers on all 4 sandbox datasets with production configurations.

**Usage:**
```bash
# Make executable (first time only)
chmod +x run_comprehensive_benchmark.sh

# Run with default 1-hour timeout per synthesizer
./run_comprehensive_benchmark.sh

# Run with custom timeout (2 hours per synthesizer)
./run_comprehensive_benchmark.sh 7200
```

**Features:**
- **Automated Processing**: Runs all 4 datasets sequentially without manual intervention
- **Smart Sample Sizes**: Automatically selects optimal sample counts per dataset
- **Comprehensive Logging**: Creates detailed logs and progress tracking
- **Error Recovery**: Continues processing even if one dataset fails
- **Organized Output**: Creates timestamped directories with all results
- **Master Summary**: Generates consolidated JSON report across all runs
- **Estimated Runtime**: 8-12 hours total (varies by hardware)

**Output Structure:**
- `comprehensive_results_YYYYMMDD_HHMMSS/` - Main results directory
- `results_[dataset_name]/` - Individual dataset results
- `comprehensive_benchmark.log` - Complete execution log  
- `master_summary.json` - Consolidated statistics
- `[dataset]_report.json` - Individual dataset reports

### `simple_demo.py` - Quick Demonstration
Lightweight demo for testing and learning synthesizer basics with built-in demo data.

**Usage:**
```bash
python simple_demo.py [--synthesizer NAME] [--samples N]
```

**Options:**
- `--synthesizer NAME`: Choose from Identity, CTGAN, TVAE, TabDDPM, CART, SMOTE, TabSyn
- `--samples N`: Number of synthetic samples (default: 100)

**Examples:**
```bash
# Quick CTGAN demo with correlated demo data
python simple_demo.py --synthesizer CTGAN

# Test TabSyn performance
python simple_demo.py --synthesizer TabSyn --samples 500

# Fast tree-based method
python simple_demo.py --synthesizer CART --samples 200
```

**Features:**
- **Built-in Demo Data**: Creates realistic dataset with correlations (age, income, education, etc.)
- **Fast Execution**: Optimized hyperparameters for quick results
- **Basic Analysis**: Shows statistical comparisons and data type preservation
- **Educational**: Perfect for understanding synthesizer capabilities and limitations

## Datasets (sandbox/)

The sandbox folder contains 4 realistic advertising/marketing datasets:

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| `conversions_all_8-1-25.csv` | 5,102 | 24 | Conversion events and user actions |
| `sponsored_ads_traffic_7-29-25.csv` | 10,984 | 70 | Sponsored ad traffic and performance |
| `dsp_impressions_7-29-25.csv` | 31,794 | 77 | Display ad impressions and targeting |
| `amazon_attributed_events_by_traffic_time_7-29-25.csv` | 4,948 | 182 | Comprehensive attribution data |

These datasets provide realistic challenges with:
- **Mixed Data Types**: Numerical, categorical, boolean, datetime
- **High Dimensionality**: Up to 182 features
- **Business Correlations**: Realistic relationships between variables
- **Scale Variety**: From 5k to 32k rows for different performance testing

### Synthesizer Performance Guide

Based on dataset characteristics:

**Fast & Reliable (< 30s):**
- Identity, CART, SMOTE

**Medium Performance (1-5 minutes):**
- CTGAN, TVAE, PATECTGAN, DPCART

**Advanced Methods (5-30 minutes):**
- BayesianNetwork, ARF, AutoDiff, TabSyn

**Specialized/Slow (30+ minutes):**
- TabDDPM, GREAT, NFlow (for large datasets)

**Problematic:**
- AIM (dependency issues - skip for now)

### Output Structure

**benchmark_results/:**
- `{synthesizer}_synthetic.csv` - Generated synthetic data
- `{synthesizer}_metadata.json` - Performance metrics and config
- `{synthesizer}_error.txt` - Error logs (if failed)
- `benchmark_report.json` - Comprehensive summary with rankings

### Tips for Best Results

1. **Start Small**: Use `--n_samples 1000` for initial testing
2. **Monitor Resources**: Some synthesizers are memory-intensive
3. **Adjust Timeout**: Use `--timeout 3600` for large/complex datasets
4. **Check Quality**: Compare synthetic vs. original data distributions
5. **Use Appropriate Methods**: Let smart selection choose optimal synthesizers

## Additional Files

- `example.ipynb` - CART synthesizer tutorial (Jupyter notebook)