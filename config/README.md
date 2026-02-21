# Model Configuration Files

This directory contains JSON configuration files for all compatible models in the table-synthesizers project.

## File Structure

Each model has a configuration file named `default_{MODEL_NAME}.json` with the following structure:

```json
{
    "model_type": "gpu" or "cpu",
    "epochs": 50,
    "batch_size": 200,
    // ... model-specific hyperparameters ...
    "description": "Model description",
    "speedup": "Estimated speedup (e.g., '15-30x')",
    "quality": "Quality rating (High/Good/Medium/Perfect)"
}
```

## GPU-Optimized Models (10-40x speedup)

### default_CTGAN.json
- **Conditional GAN** - Best balance of speed and quality
- Hyperparameters: epochs, batch_size, embedding_dim, generator_dim, discriminator_dim, learning rates

### default_TVAE.json
- **Variational Autoencoder** - Fastest training
- Hyperparameters: epochs, batch_size, embedding_dim, compress_dims, decompress_dims

### default_TabDDPM.json
- **Diffusion Model** - Highest quality
- Hyperparameters: epochs, batch_size, num_timesteps, gaussian_loss_type, scheduler

### default_PATE-CTGAN.json
- **Privacy-Preserving GAN**
- Hyperparameters: epochs, batch_size, num_teachers, epsilon, delta, embedding_dim

### default_AutoDiff.json
- **Automatic Differential Privacy**
- Hyperparameters: epochs, batch_size, training_steps, epsilon, delta

## CPU-Only Models

### default_CART.json
- **Decision Tree Synthesizer**
- No specific hyperparameters (uses sklearn defaults)

### default_DPCART.json
- **Differential Privacy Decision Tree**
- Hyperparameters: epsilon

### default_SMOTE.json
- **Oversampling Technique**
- Hyperparameters: k_neighbors

### default_Identity.json
- **Pass-through** (returns original data)
- No hyperparameters

### default_AIM.json
- **Adaptive and Iterative Mechanism**
- No specific hyperparameters (uses sklearn defaults)

## Synthcity-Based Models

These models use the [synthcity](https://github.com/vanderschaarlab/synthcity) plugin system as their backend.

### default_ARF.json
- **Adversarial Random Forest** - Combines adversarial training with random forest-based generation
- Backend: synthcity plugin `arf`

### default_BayesianNetwork.json
- **Bayesian Network** - Learns conditional dependencies between columns
- Backend: synthcity plugin `bayesian_network`

### default_GREAT.json
- **GeneRative fEAture Transformer** - Transformer-based generative model for mixed-type tabular data
- Backend: synthcity plugin `great`
- Supports GPU acceleration

### default_NFlow.json
- **Normalizing Flow** - Learns invertible transformations to model data distribution
- Backend: synthcity plugin `nflow`

## Customizing Configurations

### Method 1: Edit Existing JSON Files
Simply edit the JSON files in this directory to change default hyperparameters:

```bash
# Edit CTGAN config
nano config/default_CTGAN.json

# Change epochs to 100 instead of 50
```

### Method 2: Create Custom Config Directory
Create a new directory with modified configs:

```bash
# Create custom config directory
mkdir -p my_configs

# Copy and modify configs
cp config/default_CTGAN.json my_configs/
nano my_configs/default_CTGAN.json

# Use custom configs
python train_all_compatible_models.py \
    --dataset insurance \
    --config_dir ./my_configs \
    --models CTGAN
```

### Method 3: Override at Runtime
Use command-line arguments to override specific parameters:

```bash
# Override epochs for all models
python train_all_compatible_models.py \
    --dataset insurance \
    --epochs 100 \
    --models CTGAN TVAE

# This overrides the 'epochs' value from config files
```

## JSON Array vs Tuple Notation

**Important**: JSON uses arrays `[256, 256]` while Python code uses tuples `(256, 256)`.

The loading function automatically converts:
- `generator_dim`, `discriminator_dim`: JSON arrays → Python tuples
- `compress_dims`, `decompress_dims`: JSON arrays → Python tuples

Example:
```json
{
    "generator_dim": [256, 256]  // JSON array
}
```

Becomes in Python:
```python
config['generator_dim'] = (256, 256)  # Python tuple
```

## Batch Size Optimization

Note that `batch_size` values in config files are **default values** that will be **automatically optimized** based on:
- GPU memory availability
- Dataset size
- Model-specific constraints (e.g., CTGAN PAC divisibility)

The script will log optimization like:
```
Batch size optimized: 200 → 107
```

## Adding New Models

To add a new model configuration:

1. Create `config/default_YOUR_MODEL.json`:
```json
{
    "model_type": "gpu",
    "epochs": 50,
    "batch_size": 128,
    "your_param": "value",
    "description": "Your model description",
    "speedup": "10-20x",
    "quality": "Good"
}
```

2. Update the model list in `train_all_compatible_models.py`:
```python
all_models = ['CTGAN', 'TVAE', ..., 'YOUR_MODEL']
```

3. Implement the training logic in `TableSynthesizer` class.

## Error Handling

If a config file is missing or invalid:
- **Warning**: Script will skip that model and continue with others
- **Error**: If ALL configs fail to load, script will exit with error message

Example warning:
```
Warning: Config file not found: config/default_NEWMODEL.json, skipping NEWMODEL
```

## Config Validation

The loading function validates:
- ✅ File exists and is valid JSON
- ✅ Required fields present (model_type, description, speedup, quality)
- ✅ Proper data types (integers, floats, strings, arrays)
- ✅ Array-to-tuple conversion for dimension parameters

## Best Practices

1. **Backup before editing**: Always keep a copy of original configs
   ```bash
   cp -r config config_backup
   ```

2. **Test with minimal parameters**: Use `--epochs 1 --samples 10` to validate changes
   ```bash
   python train_all_compatible_models.py \
       --dataset insurance \
       --epochs 1 \
       --samples 10 \
       --models YOUR_MODEL
   ```

3. **Document changes**: Add comments to your custom configs (not in JSON, but in a separate notes file)

4. **Version control**: Commit config changes to git with descriptive messages
   ```bash
   git add config/
   git commit -m "Increase CTGAN epochs to 100 for better quality"
   ```

## Related Files

- **train_all_compatible_models.py**: Main training script that loads these configs
- **ENHANCED_TRAINING_GUIDE.md**: Comprehensive usage guide
- **BATCH_SIZE_OPTIMIZATION_SUMMARY.md**: Details on batch size optimization

## Questions?

See the main project documentation or open an issue on GitHub.
