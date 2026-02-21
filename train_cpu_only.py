#!/usr/bin/env python3
"""
Helper script to train a model on CPU only (no CUDA).
Used by train_all_compatible_models.py for GPU fallback.
"""
import sys
import os
import torch
import pandas as pd
import pickle
from pathlib import Path

# Ensure CUDA is disabled
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Find project root (where this script is located)
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"[CPU-ONLY] Project root: {PROJECT_ROOT}")

# Add project to path using absolute path
src_path = PROJECT_ROOT / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))
    print(f"[CPU-ONLY] Added to path: {src_path}")
else:
    print(f"[CPU-ONLY] Warning: src directory not found at {src_path}")
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))  # Fallback

from stg.tableSynthesizer import TableSynthesizer


def train_on_cpu(model_name, config, train_df, num_samples, verbose=False):
    """Train model on CPU only."""
    try:
        if verbose:
            print(f"[CPU-ONLY] PyTorch CUDA available: {torch.cuda.is_available()}")
            print(f"[CPU-ONLY] Creating TableSynthesizer for {model_name}")

        # Create synthesizer (should auto-detect CPU since CUDA is disabled)
        synthesizer = TableSynthesizer(model_name, config=config)

        # Force CPU explicitly
        synthesizer.model._device = torch.device('cpu')
        synthesizer.model.device = torch.device('cpu')

        if verbose:
            print(f"[CPU-ONLY] Device: {synthesizer.model.device}")

        # Train
        print(f"Training on {len(train_df)} samples (CPU mode)...")
        from datetime import datetime
        start_time = datetime.now()
        synthesizer.fit(train_df)
        training_time = (datetime.now() - start_time).total_seconds()

        # Generate
        print(f"Generating {num_samples} synthetic samples...")
        gen_start_time = datetime.now()
        synthetic_df = synthesizer.sample(n=num_samples, return_dataframe=True)
        generation_time = (datetime.now() - gen_start_time).total_seconds()

        print(f"✅ {model_name} completed on CPU in {training_time:.1f}s ({training_time/60:.1f} min)")

        return {
            'success': True,
            'training_time': training_time,
            'generation_time': generation_time,
            'synthetic_df': synthetic_df
        }

    except Exception as e:
        import traceback
        print(f"❌ CPU training failed: {type(e).__name__}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()[:500]}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description='Train model on CPU only')
    parser.add_argument('--model_name', required=True, help='Model name (e.g., CTGAN)')
    parser.add_argument('--config_file', required=True, help='Pickled config file')
    parser.add_argument('--train_data_file', required=True, help='Pickled training data file')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of synthetic samples')
    parser.add_argument('--output_file', required=True, help='Output pickle file')
    parser.add_argument('--cpu_cores', type=int, default=None, help='Number of CPU cores to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Configure CPU threading
    cpu_cores = args.cpu_cores if args.cpu_cores else multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(cpu_cores)
    os.environ['MKL_NUM_THREADS'] = str(cpu_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_cores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_cores)

    if args.verbose:
        print(f"[CPU-ONLY] Using {cpu_cores} CPU cores")
        print(f"[CPU-ONLY] OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

    # Load config and training data
    with open(args.config_file, 'rb') as f:
        config = pickle.load(f)

    with open(args.train_data_file, 'rb') as f:
        train_df = pickle.load(f)

    # Train on CPU
    result = train_on_cpu(args.model_name, config, train_df, args.num_samples, args.verbose)

    # Save result
    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)
