# WandB Integration Guide

**Date**: 2026-02-05
**Feature**: Weights & Biases (WandB) metrics tracking and visualization

---

## ✅ What Was Implemented

### WandB Integration Features

**Comprehensive Metrics Tracking**:
- Training time per model
- Generation time for synthetic data
- Batch size optimization (original vs optimized)
- Dataset statistics (samples, features)
- Device information (CUDA/MPS/CPU)
- Success/failure status
- Model configurations
- Per-dataset and cross-dataset summaries

**Organized Dashboards**:
- Hierarchical metric naming (`{model}/metric_name`)
- Summary tables for quick overview
- Real-time training monitoring
- Cross-run comparisons

---

## 🚀 Quick Start

### 1. Install WandB

```bash
pip install wandb
```

### 2. Configure WandB

**Option A: Environment Variables (.env file)**
```bash
# Copy example file
cp .env.example .env

# Edit with your API key
nano .env

# Add your API key
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=table-synthesizers
```

**Option B: WandB Login**
```bash
# Login interactively
wandb login

# Or set API key directly
export WANDB_API_KEY=your_api_key_here
```

### 3. Enable WandB Logging

```bash
# Train with WandB enabled
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN TVAE \
    --use_wandb
```

---

## 📖 Usage Examples

### Example 1: Single Dataset with WandB

```bash
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN TVAE \
    --epochs 50 \
    --samples 1000 \
    --use_wandb \
    --wandb_project my-experiments
```

**WandB Output:**
- Run name: `insurance_CTGAN-TVAE`
- Metrics logged: Training time, batch sizes, device info
- Tags: `insurance`, `table-synthesizers`, `CTGAN`, `TVAE`

### Example 2: Iterative Mode with WandB

```bash
python train_all_compatible_models.py \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN \
    --use_wandb \
    --wandb_project batch-synthesis
```

**WandB Output:**
- Multiple runs (one per dataset)
- Run names: `dataset1_CTGAN`, `dataset2_CTGAN`, etc.
- Cross-run comparisons possible

### Example 3: Custom WandB Project and Entity

```bash
python train_all_compatible_models.py \
    --dataset insurance \
    --models TVAE \
    --use_wandb \
    --wandb_project production-synthesis \
    --wandb_entity my-team
```

### Example 4: Using Environment Variables

```bash
# Set in .env file
WANDB_PROJECT=experiments-2024
WANDB_ENTITY=data-science-team

# Run without specifying arguments (uses env vars)
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN \
    --use_wandb
```

---

## 📊 Metrics Logged

### Per-Model Metrics

Each model logs metrics under `{model_name}/` namespace:

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `dataset_samples` | Number of training samples | 1070 |
| `dataset_features` | Number of features/columns | 7 |
| `batch_size_original` | Original batch size from config | 200 |
| `batch_size_optimized` | Optimized batch size | 107 |
| `target_synthetic_samples` | Requested synthetic samples | 1000 |
| `device_type` | Device used (1=CUDA, 0.5=MPS, 0=CPU) | 1.0 |
| `training_time_seconds` | Training duration in seconds | 120.5 |
| `training_time_minutes` | Training duration in minutes | 2.01 |
| `generation_time_seconds` | Synthetic data generation time | 2.3 |
| `synthetic_samples_generated` | Actual samples generated | 1000 |
| `samples_per_second` | Training throughput | 8.9 |
| `status` | Success (1) or failure (0) | 1 |
| `config_{param}` | Model hyperparameters | varies |

**Example Metrics for CTGAN:**
```
CTGAN/dataset_samples: 1070
CTGAN/dataset_features: 7
CTGAN/batch_size_original: 200
CTGAN/batch_size_optimized: 107
CTGAN/training_time_seconds: 120.5
CTGAN/training_time_minutes: 2.01
CTGAN/generation_time_seconds: 2.3
CTGAN/status: 1
CTGAN/config_epochs: 50
CTGAN/config_embedding_dim: 128
```

### Summary Metrics

Global metrics for the entire run:

| Metric | Description |
|--------|-------------|
| `summary/total_models` | Number of models trained |
| `summary/successful_models` | Number of successful trainings |
| `summary/failed_models` | Number of failed trainings |
| `summary/total_training_time_seconds` | Total time across all models |
| `summary/total_training_time_minutes` | Total time in minutes |
| `summary/average_time_per_model` | Average time per model |
| `summary_table` | Table with all results |

### Summary Table

WandB table with columns:
- Model name
- Model type (GPU/CPU)
- Status (success/failed)
- Training time
- Generation time

---

## 🔧 Configuration

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb` | flag | False | Enable WandB logging |
| `--wandb_project` | str | table-synthesizers | WandB project name |
| `--wandb_entity` | str | None | WandB entity/team name |

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `WANDB_API_KEY` | API key for authentication | `abc123...` |
| `WANDB_PROJECT` | Default project name | `table-synthesizers` |
| `WANDB_ENTITY` | Default entity/team | `my-team` |
| `WANDB_MODE` | Logging mode | `online`, `offline`, `disabled` |
| `WANDB_DIR` | Local WandB directory | `./wandb` |

### Priority Order

Arguments are resolved in this order (highest to lowest priority):
1. Command-line arguments (`--wandb_project`)
2. Environment variables (`WANDB_PROJECT`)
3. Default values

**Example:**
```bash
# .env file
WANDB_PROJECT=default-project

# Command-line overrides .env
python train_all_compatible_models.py \
    --use_wandb \
    --wandb_project custom-project  # This takes priority
```

---

## 📁 WandB Run Structure

### Run Configuration

Each run includes configuration metadata:

```python
config = {
    'dataset_name': 'insurance',
    'dataset_path': '/path/to/insurance.csv',
    'dataset_shape': (1338, 7),
    'train_samples': 1070,
    'test_samples': 268,
    'num_features': 7,
    'models': ['CTGAN', 'TVAE'],
    'target_samples': 1000,
    'test_size': 0.2,
    'file_type': 'csv'
}
```

### Run Naming

Runs are automatically named:
- Single dataset: `{dataset_name}_{model1}-{model2}`
- Example: `insurance_CTGAN-TVAE`

### Run Tags

Automatic tags for easy filtering:
- Dataset name (e.g., `insurance`)
- Model names (e.g., `CTGAN`, `TVAE`)
- Fixed tag: `table-synthesizers`

**Example:**
```
Tags: ['insurance', 'table-synthesizers', 'CTGAN', 'TVAE']
```

---

## 🎯 Use Cases

### 1. Compare Model Performance

Track which models train fastest:

```bash
# Train all GPU models with WandB
python train_all_compatible_models.py \
    --dataset insurance \
    --group gpu \
    --use_wandb

# View in WandB:
# - Sort by training_time_seconds
# - Compare {model}/training_time_minutes
```

### 2. Monitor Batch Size Optimization

Track optimization effectiveness:

```bash
# Run with WandB
python train_all_compatible_models.py \
    --dataset large_dataset \
    --models CTGAN \
    --use_wandb

# View in WandB:
# - CTGAN/batch_size_original vs batch_size_optimized
# - See memory optimization impact
```

### 3. Track Multi-Dataset Experiments

Compare performance across datasets:

```bash
# Iterative mode with WandB
python train_all_compatible_models.py \
    --data_folder ./datasets \
    --iterate_datasets \
    --models CTGAN TVAE \
    --use_wandb \
    --wandb_project dataset-comparison

# View in WandB:
# - Group runs by dataset
# - Compare training times across datasets
```

### 4. Production Monitoring

Track production synthetic data generation:

```bash
# Production run with WandB
python train_all_compatible_models.py \
    --dataset production_data \
    --models CTGAN \
    --epochs 100 \
    --samples 10000 \
    --use_wandb \
    --wandb_project production \
    --wandb_entity data-team
```

### 5. Hyperparameter Tuning

Track different configurations:

```bash
# Run 1: Default epochs
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN \
    --use_wandb

# Run 2: More epochs
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN \
    --epochs 100 \
    --use_wandb

# Compare in WandB dashboard
```

---

## 📈 WandB Dashboard Examples

### Metric Visualization

**Training Time Comparison:**
```
Chart: Line plot
X-axis: Run
Y-axis: {model}/training_time_minutes
Group by: Model
```

**Batch Size Optimization:**
```
Chart: Bar plot
Metrics: batch_size_original, batch_size_optimized
Compare: Side by side
```

**Success Rate:**
```
Chart: Pie chart
Metric: summary/successful_models vs summary/failed_models
```

### Custom Views

**Performance Dashboard:**
1. Training time line chart
2. Samples per second bar chart
3. Summary table
4. Device distribution pie chart

**Optimization Dashboard:**
1. Batch size before/after comparison
2. Memory usage (if logged)
3. Training throughput
4. GPU utilization (if available)

---

## 🚨 Troubleshooting

### Issue: "wandb not installed"

**Error:**
```
Warning: wandb not installed. Install with: pip install wandb
```

**Solution:**
```bash
pip install wandb
```

### Issue: "WandB initialization failed"

**Error:**
```
⚠️  WandB initialization failed: Invalid API key
```

**Solution:**
```bash
# Check API key
echo $WANDB_API_KEY

# Login again
wandb login

# Or set API key
export WANDB_API_KEY=your_actual_key
```

### Issue: "WandB run not showing metrics"

**Possible causes:**
1. `--use_wandb` flag not set
2. WandB login failed
3. Network issues (use `--wandb_mode offline`)

**Solution:**
```bash
# Check if WandB is enabled
python train_all_compatible_models.py --help | grep wandb

# Enable explicitly
python train_all_compatible_models.py \
    --dataset insurance \
    --models CTGAN \
    --use_wandb  # <-- Must include this

# Use offline mode
export WANDB_MODE=offline
python train_all_compatible_models.py --dataset insurance --use_wandb

# Sync later
wandb sync ./wandb/run-XXXXX
```

### Issue: "Too many runs created"

**Problem**: Iterative mode creates one run per dataset

**Solution:**
```bash
# Expected behavior - this is correct
# Each dataset gets its own run for comparison

# If you want fewer runs, train specific datasets:
python train_all_compatible_models.py \
    --dataset insurance \
    --use_wandb  # Single run
```

---

## ⚙️ Advanced Configuration

### Offline Mode

Save logs locally, sync later:

```bash
# Set offline mode
export WANDB_MODE=offline

# Train
python train_all_compatible_models.py \
    --dataset insurance \
    --use_wandb

# Sync when ready
wandb sync ./wandb/run-20240205_123456-abc123
```

### Custom WandB Directory

```bash
export WANDB_DIR=/custom/path/wandb

python train_all_compatible_models.py \
    --dataset insurance \
    --use_wandb
```

### Disable WandB Temporarily

```bash
# Method 1: Don't use --use_wandb flag
python train_all_compatible_models.py --dataset insurance

# Method 2: Set disabled mode
export WANDB_MODE=disabled
python train_all_compatible_models.py --dataset insurance --use_wandb
```

---

## 📝 Best Practices

### 1. Use Consistent Project Names

```bash
# Development
--wandb_project table-syn-dev

# Staging
--wandb_project table-syn-staging

# Production
--wandb_project table-syn-prod
```

### 2. Tag Runs Appropriately

WandB automatically tags with:
- Dataset name
- Model names
- `table-synthesizers`

Add custom tags via environment:
```bash
export WANDB_TAGS=experiment-1,gpu-optimized
```

### 3. Use Descriptive Run Names

Run names auto-generated as:
```
{dataset_name}_{model1}-{model2}
```

Customize by setting different dataset names.

### 4. Group Related Runs

Use same project for related experiments:
```bash
# All hyperparameter tuning in one project
--wandb_project hyperparam-tuning
```

### 5. Monitor in Real-Time

```bash
# Keep WandB dashboard open during training
# Navigate to: https://wandb.ai/{entity}/{project}
```

---

## 🎓 Integration Details

### Code Changes

**Files Modified:**
- `train_all_compatible_models.py`

**Functions Updated:**
- `train_and_generate()`: Added WandB metric logging
- `train_on_dataset()`: Added WandB run initialization and finalization
- `main()`: Added WandB command-line arguments

**New Dependencies:**
- `wandb` (optional, graceful degradation if not installed)

### Metrics Implementation

```python
# Pre-training metrics
wandb_run.log({
    f'{model_name}/dataset_samples': len(train_df),
    f'{model_name}/batch_size_optimized': config['batch_size'],
    # ... more metrics
})

# Post-training metrics
wandb_run.log({
    f'{model_name}/training_time_seconds': training_time,
    f'{model_name}/status': 1,  # Success
    # ... more metrics
})

# Summary metrics
wandb_run.log({
    'summary/total_models': len(results),
    'summary/successful_models': successful,
    # ... more metrics
})
```

---

## 📚 Related Documentation

- **ENHANCED_TRAINING_GUIDE.md**: General training guide
- **ITERATIVE_TRAINING_GUIDE.md**: Batch processing guide
- **CONFIG_EXTERNALIZATION_SUMMARY.md**: Configuration guide
- **BATCH_SIZE_OPTIMIZATION_SUMMARY.md**: Batch optimization details

---

## ✅ Summary

**Key Features:**
- ✅ Real-time metrics tracking
- ✅ Hierarchical metric organization
- ✅ Per-model and summary statistics
- ✅ Success/failure tracking
- ✅ Device and configuration logging
- ✅ Summary tables for quick overview
- ✅ Environment variable support
- ✅ Offline mode support
- ✅ Graceful degradation (optional dependency)

**When to Use WandB:**
- Comparing model performance
- Monitoring production training
- Hyperparameter tuning
- Cross-dataset experiments
- Team collaboration

**Status**: Production-ready ✅
