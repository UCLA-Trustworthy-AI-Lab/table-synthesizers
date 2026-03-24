#!/usr/bin/env python3
"""
Train all compatible models (GPU-optimized and CPU-only) on single or multiple datasets.
Designed for Python 3.12 + ARM64 Linux compatibility.

Model Groups:
1. GPU-Optimized Models (10-40x speedup):
   - CTGAN, TVAE, TabDDPM, PATE-CTGAN, AutoDiff, GREAT

2. CPU-Only Models:
   - CART, DPCART, SMOTE, Identity, AIM

3. Synthcity-Based Models:
   - ARF, BayesianNetwork, GREAT, NFlow

Single Dataset Mode:
    # Train all GPU-optimized models
    python train_all_compatible_models.py --dataset insurance --epochs 50 --samples 1000

    # Train specific models
    python train_all_compatible_models.py --dataset insurance --models CTGAN TVAE

    # Train all models (GPU + CPU)
    python train_all_compatible_models.py --dataset insurance --group all

Iterative Dataset Mode (NEW):
    # Train on all CSV files in folder
    python train_all_compatible_models.py --data_folder ./datasets --file_type csv --iterate_datasets --models CTGAN

    # Train on all Parquet files recursively
    python train_all_compatible_models.py --data_folder ./data --file_type parquet --iterate_datasets --recursive

    # Quick test on multiple datasets
    python train_all_compatible_models.py --data_folder ./test_data --iterate_datasets --epochs 1 --samples 10
"""

import argparse
import sys
import json
import os
import subprocess
import pickle
import tempfile
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# ============================================================
# Project Root Detection and Path Resolution
# ============================================================

def find_project_root():
    """
    Find the project root directory (where this script is located).

    Returns:
        Path: Absolute path to project root
    """
    # Get the directory where this script is located
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    return project_root

def resolve_path(path, project_root):
    """
    Resolve a path that may be relative or absolute.
    If relative, resolve it relative to the project root.

    Args:
        path: Path string or Path object
        project_root: Project root directory

    Returns:
        Path: Absolute path
    """
    path_obj = Path(path)

    # If already absolute, return as-is
    if path_obj.is_absolute():
        return path_obj

    # If relative, resolve relative to project root
    resolved = (project_root / path_obj).resolve()
    return resolved

# Detect project root at module load time
PROJECT_ROOT = find_project_root()
print(f"[INFO] Project root: {PROJECT_ROOT}")
print(f"[INFO] Current working directory: {Path.cwd()}")

# Add src to path using absolute path
src_path = PROJECT_ROOT / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    print(f"[WARNING] src directory not found at {src_path}")
    sys.path.insert(0, 'src')  # Fallback to relative

# Configure multi-threading for CPU training BEFORE importing synthesizer modules
# This ensures PyTorch and numerical libraries use all available cores
import multiprocessing
default_cores = multiprocessing.cpu_count()
os.environ.setdefault('OMP_NUM_THREADS', str(default_cores))
os.environ.setdefault('MKL_NUM_THREADS', str(default_cores))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(default_cores))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(default_cores))
os.environ.setdefault('TABSYN_NUM_THREADS', str(default_cores))

# Set PyTorch thread count (will be respected by synthesizer modules)
torch.set_num_threads(default_cores)

from stg.tableSynthesizer import TableSynthesizer, DEFAULT_MODELS
from stg.gpu_utils import (
    detect_best_device,
    get_device_info,
    get_optimal_batch_size,
    print_gpu_info,
    is_gpu_available
)

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def load_model_configs(config_dir='./config'):
    """
    Load model configurations from JSON files.

    Args:
        config_dir: Directory containing model config JSON files

    Returns:
        tuple: (GPU_MODEL_CONFIGS dict, CPU_MODEL_CONFIGS dict)
    """
    # Resolve config directory relative to project root
    config_path = resolve_path(config_dir, PROJECT_ROOT)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config directory not found: {config_dir}\n"
            f"  Resolved to: {config_path}\n"
            f"  Project root: {PROJECT_ROOT}\n"
            f"  Current working directory: {Path.cwd()}"
        )

    gpu_configs = {}
    cpu_configs = {}

    # List of all expected model names
    # Note: Config files may have different names (e.g., PATE-CTGAN) than model registry names (PATECTGAN)
    model_name_mapping = {
        'CTGAN': 'CTGAN',
        'TVAE': 'TVAE',
        'TabDDPM': 'TabDDPM',
        'PATE-CTGAN': 'PATECTGAN',  # Config file has hyphen, model registry doesn't
        'AutoDiff': 'AutoDiff',
        'CART': 'CART',
        'DPCART': 'DPCART',
        'SMOTE': 'SMOTE',
        'Identity': 'Identity',
        'AIM': 'AIM',
        'TabSyn': 'TabSyn',
        'ARF': 'ARF',
        'BayesianNetwork': 'BayesianNetwork',
        'GREAT': 'GREAT',
        'NFlow': 'NFlow',
    }

    for config_name, model_name in model_name_mapping.items():
        # Check if model is available in DEFAULT_MODELS
        if model_name not in DEFAULT_MODELS:
            print(f"Info: Model {model_name} not available (missing dependencies), skipping")
            continue

        config_file = config_path / f"default_{config_name}.json"

        if not config_file.exists():
            print(f"Warning: Config file not found: {config_file}, skipping {model_name}")
            continue

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Convert JSON arrays back to tuples for dimension specifications
            for key in ['generator_dim', 'discriminator_dim', 'compress_dims', 'decompress_dims']:
                if key in config and isinstance(config[key], list):
                    config[key] = tuple(config[key])

            # Separate by model type
            model_type = config.pop('model_type', 'gpu')  # Default to GPU if not specified

            if model_type == 'gpu':
                gpu_configs[model_name] = config  # Use model_name (not config_name)
            else:
                cpu_configs[model_name] = config  # Use model_name (not config_name)

        except Exception as e:
            print(f"Error loading config for {model_name}: {e}")
            continue

    if not gpu_configs and not cpu_configs:
        raise ValueError("No valid model configurations loaded from config directory")

    return gpu_configs, cpu_configs


# Load model configurations from JSON files (default location)
# These can be overridden in main() if --config_dir is provided
try:
    GPU_MODEL_CONFIGS, CPU_MODEL_CONFIGS = load_model_configs()
except Exception as e:
    print(f"Warning: Could not load default configs from ./config: {e}")
    print("Configs will need to be loaded from custom location via --config_dir")
    GPU_MODEL_CONFIGS, CPU_MODEL_CONFIGS = {}, {}


def discover_datasets(data_folder, file_type='csv', recursive=False):
    """
    Discover all datasets in a folder.

    Args:
        data_folder: Path to folder containing datasets
        file_type: File extension to search for ('csv' or 'parquet')
        recursive: If True, search subdirectories recursively

    Returns:
        list: List of tuples (dataset_name, file_path)
    """
    # Resolve data folder path
    data_path = resolve_path(data_folder, PROJECT_ROOT)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_folder}\n"
            f"  Resolved to: {data_path}\n"
            f"  Project root: {PROJECT_ROOT}\n"
            f"  Current working directory: {Path.cwd()}"
        )

    if not data_path.is_dir():
        raise ValueError(
            f"Data folder must be a directory: {data_folder}\n"
            f"  Resolved to: {data_path}"
        )

    # Search pattern
    if recursive:
        pattern = f"**/*.{file_type}"
    else:
        pattern = f"*.{file_type}"

    # Find all matching files
    dataset_files = list(data_path.glob(pattern))

    if not dataset_files:
        search_desc = "recursively" if recursive else "in directory"
        raise ValueError(
            f"No {file_type} files found {search_desc}: {data_folder}\n"
            f"  Searched in: {data_path}\n"
            f"  Pattern: {pattern}"
        )

    # Create list of (dataset_name, file_path) tuples
    # file_path from glob is already absolute
    datasets = []
    for file_path in sorted(dataset_files):
        # Use filename without extension as dataset name
        dataset_name = file_path.stem
        datasets.append((dataset_name, str(file_path)))

    return datasets


def print_device_info():
    """Print GPU information if available."""
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)

    device_info = get_device_info()
    print(f"Device Type: {device_info['type']}")

    if device_info['type'] == 'cuda':
        print(f"GPU Name: {device_info['name']}")
        print(f"Total Memory: {device_info['memory_gb']:.1f} GB")
        print(f"Compute Capability: {device_info['compute_capability'][0]}.{device_info['compute_capability'][1]}")
        print(f"CUDA Version: {device_info['cuda_version']}")
    elif device_info['type'] == 'mps':
        print(f"Device: {device_info['name']}")
    else:
        print("No GPU available - using CPU")

    print("="*60 + "\n")


def load_dataset(data_path, dataset_name=None):
    """Load dataset from file or folder."""
    data_path = Path(data_path)

    if data_path.is_file():
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
    elif data_path.is_dir():
        if dataset_name:
            csv_file = data_path / f"{dataset_name}.csv"
        else:
            # Find first CSV file
            csv_files = list(data_path.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in {data_path}")
            csv_file = csv_files[0]

        print(f"Loading dataset from: {csv_file}")
        df = pd.read_csv(csv_file)
    else:
        raise ValueError(f"Invalid data path: {data_path}")

    return df


def get_model_config(model_name, override_epochs=None):
    """Get default configuration for a model."""
    # Check GPU models first
    if model_name in GPU_MODEL_CONFIGS:
        config = GPU_MODEL_CONFIGS[model_name].copy()
    elif model_name in CPU_MODEL_CONFIGS:
        config = CPU_MODEL_CONFIGS[model_name].copy()
    else:
        # Unknown model, return minimal config
        config = {'epochs': override_epochs or 50}
        return config

    # Remove metadata fields (not model parameters)
    config.pop('description', None)
    config.pop('speedup', None)
    config.pop('quality', None)
    config.pop('backend', None)

    # Override epochs if specified
    if override_epochs is not None:
        if 'epochs' in config:
            config['epochs'] = override_epochs
        # Also override diff_n_epochs for AutoDiff (diffusion stage epochs)
        if 'diff_n_epochs' in config:
            config['diff_n_epochs'] = override_epochs

    return config


def optimize_batch_size(model_name, dataset_size, default_batch_size):
    """
    Calculate optimal batch size using GPU utilities.

    Args:
        model_name: Name of the model
        dataset_size: Number of samples in dataset
        default_batch_size: Default batch size from model config

    Returns:
        Optimized batch size
    """
    # Only optimize for GPU models
    if model_name not in GPU_MODEL_CONFIGS:
        return default_batch_size

    # Use GPU utility to calculate optimal batch size
    if is_gpu_available():
        # Get optimal batch size based on GPU memory
        optimal_batch_size = get_optimal_batch_size(
            dataset_size=dataset_size,
            model_memory_per_sample_mb=1.0,  # Conservative estimate
            default_batch_size=default_batch_size
        )

        # For CTGAN, ensure batch size is divisible by PAC size
        if model_name == 'CTGAN':
            config = GPU_MODEL_CONFIGS.get(model_name, {})
            pac_size = config.get('pac', 5)
            optimal_batch_size = (optimal_batch_size // pac_size) * pac_size
            if optimal_batch_size == 0:
                optimal_batch_size = pac_size

        return optimal_batch_size
    else:
        # CPU - use smaller batch size
        return min(default_batch_size, max(32, dataset_size // 10))


def train_and_generate(model_name, train_df, config, num_samples, is_gpu_model=True,
                       use_wandb=False, dataset_name=None, wandb_run=None, verbose=False, cpu_cores=None):
    """Train model and generate synthetic samples with optional WandB logging."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Print model info
    if model_name in GPU_MODEL_CONFIGS:
        info = GPU_MODEL_CONFIGS[model_name]
        print(f"Type: GPU-optimized ({info['speedup']} speedup)")
        print(f"Quality: {info['quality']}")
        print(f"Description: {info['description']}")
    elif model_name in CPU_MODEL_CONFIGS:
        info = CPU_MODEL_CONFIGS[model_name]
        print(f"Type: CPU-only")
        print(f"Description: {info['description']}")

    # Get device info for logging
    device_info = get_device_info()
    device_type = device_info['type']

    if verbose:
        print(f"\n[VERBOSE] PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[VERBOSE] PyTorch CUDA device count: {torch.cuda.device_count()}")
            print(f"[VERBOSE] PyTorch CUDA current device: {torch.cuda.current_device()}")
            print(f"[VERBOSE] PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"[VERBOSE] PyTorch version: {torch.__version__}")
        print(f"[VERBOSE] Device info type: {device_type}")
        print(f"[VERBOSE] Is GPU model: {is_gpu_model}")

    try:
        # Optimize batch size BEFORE creating the synthesizer so the
        # constructor receives the final value.
        original_batch_size = config.get('batch_size', None)
        if 'batch_size' in config and is_gpu_model:
            optimized_batch_size = optimize_batch_size(
                model_name,
                len(train_df),
                original_batch_size
            )

            if optimized_batch_size != original_batch_size:
                config['batch_size'] = optimized_batch_size
                print(f"Batch size optimized: {original_batch_size} → {optimized_batch_size}")
            else:
                print(f"Batch size: {optimized_batch_size}")

        # Create synthesizer (with already-optimized config)
        if verbose:
            print(f"[VERBOSE] Creating TableSynthesizer for {model_name}")
        synthesizer = TableSynthesizer(model_name, config=config)

        if verbose:
            print(f"[VERBOSE] Synthesizer created: {type(synthesizer.model)}")
            print(f"[VERBOSE] Has set_device method: {hasattr(synthesizer.model, 'set_device')}")

        # Set device for GPU models
        if is_gpu_model and hasattr(synthesizer.model, 'set_device'):
            device = detect_best_device()
            if verbose:
                print(f"[VERBOSE] Detected best device: {device}")
                print(f"[VERBOSE] Calling synthesizer.model.set_device({device})")

            synthesizer.model.set_device(device)
            print(f"Device: {device}")

            if verbose:
                # Check actual device after setting
                if hasattr(synthesizer.model, '_device'):
                    print(f"[VERBOSE] Model._device after set_device: {synthesizer.model._device}")
                if hasattr(synthesizer.model, 'device'):
                    print(f"[VERBOSE] Model.device after set_device: {synthesizer.model.device}")

                # Check if generator/discriminator moved to GPU
                if hasattr(synthesizer.model, '_generator') and synthesizer.model._generator is not None:
                    gen_device = next(synthesizer.model._generator.parameters()).device
                    print(f"[VERBOSE] Generator device: {gen_device}")
                if hasattr(synthesizer.model, 'discriminator') and synthesizer.model.discriminator is not None:
                    disc_device = next(synthesizer.model.discriminator.parameters()).device
                    print(f"[VERBOSE] Discriminator device: {disc_device}")

        # Print final config
        config_str = ', '.join([f"{k}={v}" for k, v in config.items()
                               if k not in ['description', 'speedup', 'quality']])
        print(f"Config: {config_str}")

        # Log pre-training metrics to WandB
        if use_wandb and wandb_run:
            wandb_run.log({
                f'{model_name}/dataset_samples': len(train_df),
                f'{model_name}/dataset_features': len(train_df.columns),
                f'{model_name}/batch_size_original': original_batch_size or 0,
                f'{model_name}/batch_size_optimized': config.get('batch_size', 0),
                f'{model_name}/target_synthetic_samples': num_samples,
                f'{model_name}/device_type': 1 if device_type == 'cuda' else (0.5 if device_type == 'mps' else 0),
            })

        # Train
        print(f"Training on {len(train_df)} samples...")

        if verbose:
            print(f"[VERBOSE] About to call synthesizer.fit()")
            print(f"[VERBOSE] Training DataFrame shape: {train_df.shape}")

        start_time = datetime.now()
        synthesizer.fit(train_df)
        training_time = (datetime.now() - start_time).total_seconds()

        if verbose:
            print(f"[VERBOSE] Training completed in {training_time:.1f}s")
            # Check GPU utilization after training
            if torch.cuda.is_available():
                print(f"[VERBOSE] GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"[VERBOSE] GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

        # Generate samples
        print(f"Generating {num_samples} synthetic samples...")
        gen_start_time = datetime.now()
        synthetic_df = synthesizer.sample(n=num_samples, return_dataframe=True)
        generation_time = (datetime.now() - gen_start_time).total_seconds()

        if verbose:
            print(f"[VERBOSE] Generation completed in {generation_time:.1f}s")
            print(f"[VERBOSE] Synthetic data shape: {synthetic_df.shape}")

        print(f"✅ {model_name} completed in {training_time:.1f}s ({training_time/60:.1f} min)")

        # Log post-training metrics to WandB
        if use_wandb and wandb_run:
            wandb_run.log({
                f'{model_name}/training_time_seconds': training_time,
                f'{model_name}/training_time_minutes': training_time / 60,
                f'{model_name}/generation_time_seconds': generation_time,
                f'{model_name}/synthetic_samples_generated': len(synthetic_df),
                f'{model_name}/samples_per_second': len(train_df) / training_time if training_time > 0 else 0,
                f'{model_name}/status': 1,  # Success
            })

            # Log configuration
            for key, value in config.items():
                if key not in ['description', 'speedup', 'quality'] and isinstance(value, (int, float)):
                    wandb_run.log({f'{model_name}/config_{key}': value})

        return {
            'model': model_name,
            'model_type': 'GPU' if is_gpu_model else 'CPU',
            'status': 'success',
            'training_time': training_time,
            'generation_time': generation_time,
            'config': config,
            'synthetic_df': synthetic_df
        }

    except Exception as e:
        import traceback
        error_msg = str(e)

        # Check if it's a CUDA kernel error (Blackwell sm_121 compatibility issue)
        if is_gpu_model and ("no kernel image is available" in error_msg or
                             "CUDA error" in error_msg):
            print(f"\n⚠️  {model_name} failed on GPU: {type(e).__name__}: {error_msg}")
            print(f"🔄 Retrying with CPU fallback...")

            # Log GPU failure to WandB
            if use_wandb and wandb_run:
                wandb_run.log({
                    f'{model_name}/gpu_fallback': 1,
                    f'{model_name}/gpu_error': error_msg[:200],
                })

            # Retry with CPU using subprocess (clean environment)
            try:
                print(f"\n{'='*60}")
                print(f"Training {model_name} on CPU (GPU fallback via subprocess)")
                print(f"{'='*60}")

                # Clear CUDA cache completely
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if verbose:
                        print(f"[VERBOSE] Cleared CUDA cache")

                # Delete old synthesizer to free CUDA memory
                del synthesizer
                import gc
                gc.collect()

                # Use subprocess with CUDA disabled from the start
                # Create temp files for data exchange
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                    config_file = f.name
                    pickle.dump(config, f)

                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                    train_data_file = f.name
                    pickle.dump(train_df, f)

                output_file = tempfile.mktemp(suffix='.pkl')

                try:
                    # Determine CPU cores to use
                    if cpu_cores is None:
                        import multiprocessing
                        cpu_cores = multiprocessing.cpu_count()

                    # Build command (use absolute path to train_cpu_only.py)
                    train_cpu_script = PROJECT_ROOT / 'train_cpu_only.py'
                    cmd = [
                        sys.executable,
                        str(train_cpu_script),
                        '--model_name', model_name,
                        '--config_file', config_file,
                        '--train_data_file', train_data_file,
                        '--num_samples', str(num_samples),
                        '--output_file', output_file,
                        '--cpu_cores', str(cpu_cores)
                    ]
                    if verbose:
                        cmd.append('--verbose')

                    # Run subprocess with CUDA disabled and CPU threading configured
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = '-1'

                    # Set threading environment variables for CPU performance
                    env['OMP_NUM_THREADS'] = str(cpu_cores)
                    env['MKL_NUM_THREADS'] = str(cpu_cores)
                    env['NUMEXPR_NUM_THREADS'] = str(cpu_cores)
                    env['OPENBLAS_NUM_THREADS'] = str(cpu_cores)

                    if verbose:
                        print(f"[VERBOSE] Launching CPU-only subprocess with {cpu_cores} cores...")

                    result = subprocess.run(
                        cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1 hour timeout
                    )

                    # Print subprocess output
                    if result.stdout:
                        print(result.stdout, end='')
                    if result.stderr:
                        print(result.stderr, end='', file=sys.stderr)

                    # Load result
                    if result.returncode == 0 and os.path.exists(output_file):
                        with open(output_file, 'rb') as f:
                            cpu_result = pickle.load(f)

                        if cpu_result['success']:
                            training_time = cpu_result['training_time']
                            generation_time = cpu_result['generation_time']
                            synthetic_df = cpu_result['synthetic_df']
                        else:
                            raise Exception(cpu_result['error'])
                    else:
                        raise Exception(f"Subprocess failed with return code {result.returncode}")

                finally:
                    # Clean up temp files
                    for temp_file in [config_file, train_data_file, output_file]:
                        if os.path.exists(temp_file):
                            try:
                                os.unlink(temp_file)
                            except:
                                pass

                # Log CPU success to WandB
                if use_wandb and wandb_run:
                    wandb_run.log({
                        f'{model_name}/training_time_seconds': training_time,
                        f'{model_name}/training_time_minutes': training_time / 60,
                        f'{model_name}/generation_time_seconds': generation_time,
                        f'{model_name}/device_type': 0,  # CPU
                        f'{model_name}/status': 1,  # Success
                    })

                return {
                    'model': model_name,
                    'model_type': 'CPU (GPU fallback)',
                    'status': 'success',
                    'training_time': training_time,
                    'generation_time': generation_time,
                    'config': config,
                    'synthetic_df': synthetic_df
                }

            except Exception as cpu_error:
                # Restore CUDA_VISIBLE_DEVICES even on failure
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                else:
                    if 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']

                print(f"❌ {model_name} also failed on CPU: {type(cpu_error).__name__}: {str(cpu_error)}")
                print(f"Traceback: {traceback.format_exc()[:500]}")

                if use_wandb and wandb_run:
                    wandb_run.log({
                        f'{model_name}/status': 0,
                        f'{model_name}/cpu_error': str(cpu_error)[:200],
                    })

                return {
                    'model': model_name,
                    'model_type': 'GPU+CPU',
                    'status': 'failed',
                    'error': f"GPU: {error_msg}, CPU: {str(cpu_error)}"
                }

        # Non-GPU errors or CPU-only model failures
        print(f"❌ {model_name} failed: {type(e).__name__}: {error_msg}")
        print(f"Traceback: {traceback.format_exc()[:500]}")

        # Log failure to WandB
        if use_wandb and wandb_run:
            wandb_run.log({
                f'{model_name}/status': 0,  # Failure
                f'{model_name}/error': error_msg[:200],  # Truncate long errors
            })

        return {
            'model': model_name,
            'model_type': 'GPU' if is_gpu_model else 'CPU',
            'status': 'failed',
            'error': error_msg
        }


def train_on_dataset(dataset_name, dataset_path, models_to_train, args, output_base_dir):
    """
    Train models on a single dataset.

    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset file
        models_to_train: List of model names to train
        args: Command-line arguments
        output_base_dir: Base output directory

    Returns:
        dict: Training results summary
    """
    # Create dataset-specific output directory
    output_dir = Path(output_base_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"File: {dataset_path}")

    # Load dataset
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported file type: {dataset_path}")

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Split train/test
    print(f"\nSplitting data: {(1-args.test_size)*100:.0f}% train, {args.test_size*100:.0f}% test")
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Save splits
    train_df.to_csv(output_dir / 'train_data.csv', index=False)
    test_df.to_csv(output_dir / 'test_data.csv', index=False)
    print(f"Saved train/test splits to {output_dir}")

    # Initialize WandB run for this dataset
    wandb_run = None
    if args.use_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{dataset_name}_{'-'.join(models_to_train)}",
                config={
                    'dataset_name': dataset_name,
                    'dataset_path': dataset_path,
                    'dataset_shape': df.shape,
                    'train_samples': len(train_df),
                    'test_samples': len(test_df),
                    'num_features': len(df.columns),
                    'models': models_to_train,
                    'target_samples': args.samples,
                    'test_size': args.test_size,
                    'file_type': args.file_type,
                },
                tags=[dataset_name, 'table-synthesizers'] + models_to_train,
                reinit=True  # Allow multiple runs in same process
            )
            print(f"✅ WandB run initialized: {wandb_run.name}")
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
            wandb_run = None

    # Train models
    results = []

    for model_name in models_to_train:
        # Get default config for model
        config = get_model_config(model_name, override_epochs=args.epochs)

        # Determine if GPU model
        is_gpu_model = model_name in GPU_MODEL_CONFIGS

        # Train and generate
        result = train_and_generate(
            model_name,
            train_df,
            config,
            args.samples,
            is_gpu_model=is_gpu_model,
            use_wandb=args.use_wandb and WANDB_AVAILABLE,
            dataset_name=dataset_name,
            wandb_run=wandb_run,
            verbose=args.verbose,
            cpu_cores=args.cpu_cores
        )
        results.append(result)

        # Save synthetic data if successful
        if result['status'] == 'success':
            output_file = output_dir / f"synthetic_{model_name.lower()}_{args.samples}.csv"
            result['synthetic_df'].to_csv(output_file, index=False)
            print(f"Saved synthetic data to: {output_file}")

    # Print summary for this dataset
    print(f"\n{'='*60}")
    print(f"Training Summary - {dataset_name}")
    print(f"{'='*60}")

    # Group by type
    gpu_results = [r for r in results if r['model_type'] == 'GPU']
    cpu_results = [r for r in results if r['model_type'] == 'CPU']

    if gpu_results:
        print("\nGPU-Optimized Models:")
        for result in gpu_results:
            if result['status'] == 'success':
                print(f"  ✅ {result['model']:15} - {result['training_time']:7.1f}s ({result['training_time']/60:5.1f} min)")
            else:
                print(f"  ❌ {result['model']:15} - Failed")

    if cpu_results:
        print("\nCPU-Only Models:")
        for result in cpu_results:
            if result['status'] == 'success':
                print(f"  ✅ {result['model']:15} - {result['training_time']:7.1f}s ({result['training_time']/60:5.1f} min)")
            else:
                print(f"  ❌ {result['model']:15} - Failed")

    # Overall stats
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    total_time = sum(r['training_time'] for r in results if r['status'] == 'success')

    print(f"\nOverall: {successful}/{len(results)} successful, {failed} failed")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save detailed summary
    summary = []
    for result in results:
        if result['status'] == 'success':
            summary.append({
                'dataset': dataset_name,
                'model': result['model'],
                'model_type': result['model_type'],
                'status': 'success',
                'training_time_seconds': result['training_time'],
                'training_time_minutes': result['training_time'] / 60,
                'samples_generated': args.samples,
                'config': str(result['config'])
            })
        else:
            summary.append({
                'dataset': dataset_name,
                'model': result['model'],
                'model_type': result['model_type'],
                'status': 'failed',
                'error': result.get('error', 'Unknown error')
            })

    summary_df = pd.DataFrame(summary)
    summary_file = output_dir / f'training_summary_{dataset_name}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved training summary to: {summary_file}")

    # Save config reference
    config_file = output_dir / 'model_configs_reference.txt'
    with open(config_file, 'w') as f:
        f.write("GPU-Optimized Models Default Configurations\n")
        f.write("=" * 60 + "\n\n")
        for model_name, config in GPU_MODEL_CONFIGS.items():
            f.write(f"{model_name}:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("\nCPU-Only Models Default Configurations\n")
        f.write("=" * 60 + "\n\n")
        for model_name, config in CPU_MODEL_CONFIGS.items():
            f.write(f"{model_name}:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    print(f"Saved model configs reference to: {config_file}")

    # Log summary metrics to WandB
    if wandb_run:
        try:
            wandb_run.log({
                'summary/total_models': len(results),
                'summary/successful_models': successful,
                'summary/failed_models': failed,
                'summary/total_training_time_seconds': total_time,
                'summary/total_training_time_minutes': total_time / 60,
                'summary/average_time_per_model': total_time / len(results) if results else 0,
            })

            # Log summary table
            summary_data = []
            for result in results:
                if result['status'] == 'success':
                    summary_data.append([
                        result['model'],
                        result['model_type'],
                        'success',
                        f"{result['training_time']:.2f}s",
                        f"{result.get('generation_time', 0):.2f}s"
                    ])
                else:
                    summary_data.append([
                        result['model'],
                        result['model_type'],
                        'failed',
                        'N/A',
                        'N/A'
                    ])

            wandb_run.log({
                "summary_table": wandb.Table(
                    columns=["Model", "Type", "Status", "Training Time", "Generation Time"],
                    data=summary_data
                )
            })

            print(f"✅ WandB metrics logged successfully")
        except Exception as e:
            print(f"⚠️  WandB logging failed: {e}")

        # Finish WandB run
        try:
            wandb_run.finish()
            print(f"✅ WandB run finished")
        except Exception as e:
            print(f"⚠️  WandB finish failed: {e}")

    return {
        'dataset': dataset_name,
        'results': results,
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'output_dir': str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train all compatible models (GPU-optimized and CPU-only) on dataset(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single Dataset Examples:
  # Train all GPU-optimized models (default)
  python train_all_compatible_models.py --dataset insurance --epochs 50 --samples 1000

  # Train specific models
  python train_all_compatible_models.py --dataset insurance --models CTGAN TVAE

  # Train all models (GPU + CPU)
  python train_all_compatible_models.py --dataset insurance --group all

  # Train only CPU models
  python train_all_compatible_models.py --dataset insurance --group cpu

Iterative Dataset Examples (NEW):
  # Train on all CSV files in folder
  python train_all_compatible_models.py --data_folder ./datasets --file_type csv --iterate_datasets --models CTGAN TVAE

  # Train on all Parquet files recursively
  python train_all_compatible_models.py --data_folder ./data --file_type parquet --iterate_datasets --recursive --group gpu

  # Quick test on multiple datasets
  python train_all_compatible_models.py --data_folder ./test_data --iterate_datasets --models TVAE --epochs 1 --samples 10

WandB Integration Examples (NEW):
  # Enable WandB logging
  python train_all_compatible_models.py --dataset insurance --models CTGAN --use_wandb

  # WandB with custom project
  python train_all_compatible_models.py --dataset insurance --models CTGAN --use_wandb --wandb_project my-experiments

  # WandB with iterative mode
  python train_all_compatible_models.py --data_folder ./datasets --iterate_datasets --use_wandb
        """
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        default='/home/ohsono/dataset/input_data/',
        help='Path to data folder or CSV file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name (e.g., insurance) if using data_folder'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides model defaults)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of synthetic samples to generate'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2 for 80/20 split)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory for synthetic data'
    )
    parser.add_argument(
        '--group',
        type=str,
        choices=['gpu', 'cpu', 'all'],
        default='gpu',
        help='Model group to train (default: gpu)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to train (overrides --group)'
    )
    parser.add_argument(
        '--show_device_info',
        action='store_true',
        help='Print detailed GPU/device information'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='./config',
        help='Directory containing model configuration JSON files (default: ./config)'
    )
    parser.add_argument(
        '--file_type',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
        help='File type to search for when iterating datasets (default: csv)'
    )
    parser.add_argument(
        '--iterate_datasets',
        action='store_true',
        help='Train on all datasets found in data_folder (ignores --dataset)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search for datasets recursively in subdirectories'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Enable WandB logging for metrics tracking'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=os.getenv('WANDB_PROJECT', 'table-synthesizers'),
        help='WandB project name (default: table-synthesizers or $WANDB_PROJECT)'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=os.getenv('WANDB_ENTITY', None),
        help='WandB entity/team name (default: $WANDB_ENTITY)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging for debugging GPU/device issues'
    )
    parser.add_argument(
        '--cpu_cores',
        type=int,
        default=None,
        help='Number of CPU cores to use for CPU fallback (default: all available cores)'
    )

    args = parser.parse_args()

    # Load configs from custom directory if specified
    global GPU_MODEL_CONFIGS, CPU_MODEL_CONFIGS
    if args.config_dir != './config':
        print(f"Loading model configs from custom directory: {args.config_dir}")
        try:
            GPU_MODEL_CONFIGS, CPU_MODEL_CONFIGS = load_model_configs(args.config_dir)
            print(f"Loaded {len(GPU_MODEL_CONFIGS)} GPU models and {len(CPU_MODEL_CONFIGS)} CPU models")
        except Exception as e:
            print(f"Error loading configs from {args.config_dir}: {e}")
            sys.exit(1)
    elif not GPU_MODEL_CONFIGS and not CPU_MODEL_CONFIGS:
        # Default configs failed to load at module level, try again
        print("Loading model configs from default directory: ./config")
        try:
            GPU_MODEL_CONFIGS, CPU_MODEL_CONFIGS = load_model_configs(args.config_dir)
            print(f"Loaded {len(GPU_MODEL_CONFIGS)} GPU models and {len(CPU_MODEL_CONFIGS)} CPU models")
        except Exception as e:
            print(f"Error loading configs: {e}")
            sys.exit(1)

    # Print device information
    if args.show_device_info:
        print_gpu_info()
    else:
        print_device_info()

    # Determine which models to train
    if args.models:
        # Specific models requested - map config names to model names
        model_name_mapping = {
            'CTGAN': 'CTGAN',
            'ctgan': 'CTGAN',
            'TVAE': 'TVAE',
            'tvae': 'TVAE',
            'TabDDPM': 'TabDDPM',
            'tabddpm': 'TabDDPM',
            'PATE-CTGAN': 'PATECTGAN',
            'PATECTGAN': 'PATECTGAN',
            'pate-ctgan': 'PATECTGAN',
            'patectgan': 'PATECTGAN',
            'AutoDiff': 'AutoDiff',
            'autodiff': 'AutoDiff',
            'CART': 'CART',
            'cart': 'CART',
            'DPCART': 'DPCART',
            'dpcart': 'DPCART',
            'SMOTE': 'SMOTE',
            'smote': 'SMOTE',
            'Identity': 'Identity',
            'identity': 'Identity',
            'AIM': 'AIM',
            'aim': 'AIM',
            'TabSyn': 'TabSyn',
            'tabsyn': 'TabSyn',
            'ARF': 'ARF',
            'arf': 'ARF',
            'BayesianNetwork': 'BayesianNetwork',
            'bayesiannetwork': 'BayesianNetwork',
            'BaysianNetwork': 'BayesianNetwork',  # Common typo
            'BN': 'BayesianNetwork',
            'bn': 'BayesianNetwork',
            'GREAT': 'GREAT',
            'great': 'GREAT',
            'GreaT': 'GREAT',
            'Great': 'GREAT',
            'NFlow': 'NFlow',
            'nflow': 'NFlow',
            # 'LTM_VAE': 'LTM_VAE',
            # 'ltm_vae': 'LTM_VAE',
        }

        models_to_train = []
        for model_input in args.models:
            # Map config name to model name (case-insensitive fallback)
            model_name = model_name_mapping.get(model_input) or model_name_mapping.get(model_input.lower(), model_input)

            # Check if model is available
            if model_name not in DEFAULT_MODELS:
                # Provide actionable diagnostics
                synthcity_models = {'GREAT', 'ARF', 'NFlow', 'BayesianNetwork'}
                gpu_models = {'CTGAN', 'TVAE', 'TabDDPM', 'PATECTGAN', 'AutoDiff', 'TabSyn', 'LTM_VAE'}
                if model_name in synthcity_models:
                    hint = "Install synthcity: pip install -r requirements-synthcity.txt"
                    # Try to get the actual import error
                    try:
                        __import__(f"stg.{model_name}")
                    except Exception as import_err:
                        hint += f"\n       Import error: {import_err}"
                elif model_name in gpu_models:
                    hint = "Install torch: pip install -r requirements-gpu.txt (or requirements-cpu.txt)"
                else:
                    hint = f"Model '{model_name}' is not a recognized model name"
                print(f"Warning: Model {model_input} (maps to {model_name}) not available, skipping")
                print(f"  Hint: {hint}")
                continue

            # Check if we have config for this model
            if model_name not in GPU_MODEL_CONFIGS and model_name not in CPU_MODEL_CONFIGS:
                print(f"Warning: No configuration found for {model_input} (maps to {model_name}), skipping")
                continue

            models_to_train.append(model_name)

        if not models_to_train:
            print("Error: No valid models to train after filtering")
            sys.exit(1)

        print(f"Training specific models: {', '.join(models_to_train)}")
    else:
        # Train by group
        if args.group == 'gpu':
            models_to_train = list(GPU_MODEL_CONFIGS.keys())
            print(f"Training GPU-optimized models: {', '.join(models_to_train)}")
        elif args.group == 'cpu':
            models_to_train = list(CPU_MODEL_CONFIGS.keys())
            print(f"Training CPU-only models: {', '.join(models_to_train)}")
        else:  # all
            models_to_train = list(GPU_MODEL_CONFIGS.keys()) + list(CPU_MODEL_CONFIGS.keys())
            print(f"Training all models ({len(GPU_MODEL_CONFIGS)} GPU + {len(CPU_MODEL_CONFIGS)} CPU)")

    # Create output directory (resolve relative to project root)
    output_base_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_base_dir}")

    # Determine datasets to process
    if args.iterate_datasets:
        # Iterative mode: discover all datasets in folder
        print(f"\n{'='*60}")
        print("Iterative Dataset Mode")
        print(f"{'='*60}")
        print(f"Searching for {args.file_type} files in: {args.data_folder}")
        if args.recursive:
            print("Searching recursively in subdirectories")

        try:
            datasets = discover_datasets(args.data_folder, args.file_type, args.recursive)
            print(f"\nFound {len(datasets)} dataset(s):")
            for i, (name, path) in enumerate(datasets, 1):
                print(f"  {i}. {name} ({path})")
        except Exception as e:
            print(f"Error discovering datasets: {e}")
            sys.exit(1)

    else:
        # Single dataset mode
        if not args.dataset:
            print("Error: --dataset is required when not using --iterate_datasets")
            sys.exit(1)

        # Resolve data folder path
        dataset_path = resolve_path(args.data_folder, PROJECT_ROOT)

        if dataset_path.is_file():
            # Direct file path provided
            datasets = [(args.dataset, str(dataset_path))]
        else:
            # Folder + dataset name
            dataset_file = dataset_path / f"{args.dataset}.{args.file_type}"
            if not dataset_file.exists():
                print(
                    f"Error: Dataset file not found: {args.data_folder}/{args.dataset}.{args.file_type}\n"
                    f"  Resolved to: {dataset_file}\n"
                    f"  Project root: {PROJECT_ROOT}\n"
                    f"  Current working directory: {Path.cwd()}"
                )
                sys.exit(1)
            datasets = [(args.dataset, str(dataset_file))]

    # Train on each dataset
    all_results = []
    total_datasets = len(datasets)

    for idx, (dataset_name, dataset_path) in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}")
        print(f"{'#'*60}")

        try:
            dataset_result = train_on_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                models_to_train=models_to_train,
                args=args,
                output_base_dir=output_base_dir
            )
            all_results.append(dataset_result)

        except Exception as e:
            print(f"\n❌ Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'dataset': dataset_name,
                'results': [],
                'successful': 0,
                'failed': len(models_to_train),
                'total_time': 0,
                'output_dir': None,
                'error': str(e)
            })

    # Print overall summary for all datasets
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY - ALL DATASETS")
    print(f"{'='*60}")

    for dataset_result in all_results:
        dataset_name = dataset_result['dataset']
        if 'error' in dataset_result:
            print(f"\n❌ {dataset_name}: Failed - {dataset_result['error']}")
        else:
            print(f"\n✅ {dataset_name}:")
            print(f"   Models: {dataset_result['successful']}/{dataset_result['successful']+dataset_result['failed']} successful")
            print(f"   Time: {dataset_result['total_time']:.1f}s ({dataset_result['total_time']/60:.1f} min)")
            print(f"   Output: {dataset_result['output_dir']}")

    # Save combined summary
    if len(all_results) > 1:
        combined_summary = []
        for dataset_result in all_results:
            if 'results' in dataset_result and dataset_result['results']:
                for model_result in dataset_result['results']:
                    if model_result['status'] == 'success':
                        combined_summary.append({
                            'dataset': dataset_result['dataset'],
                            'model': model_result['model'],
                            'model_type': model_result['model_type'],
                            'status': 'success',
                            'training_time_seconds': model_result['training_time'],
                            'training_time_minutes': model_result['training_time'] / 60,
                            'samples_generated': args.samples,
                            'config': str(model_result['config'])
                        })

        if combined_summary:
            combined_df = pd.DataFrame(combined_summary)
            combined_file = output_base_dir / 'combined_training_summary.csv'
            combined_df.to_csv(combined_file, index=False)
            print(f"\n📊 Saved combined summary to: {combined_file}")

    # Overall statistics
    total_successful = sum(r['successful'] for r in all_results)
    total_failed = sum(r['failed'] for r in all_results)
    total_time = sum(r['total_time'] for r in all_results)

    print(f"\n{'='*60}")
    print(f"Total Datasets: {total_datasets}")
    print(f"Total Models Trained: {total_successful} successful, {total_failed} failed")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hours)")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
