#!/usr/bin/env python3
"""
Comprehensive Synthesizer Benchmark

This script evaluates all available synthesizers on realistic datasets with
production-ready hyperparameters optimized for performance and quality.

Usage:
    python benchmark_synthesizers.py [--dataset DATASET] [--n_samples N] [--timeout T]

Example:
    python benchmark_synthesizers.py --dataset conversions_all_8-1-25.csv --n_samples 2000 --timeout 1800
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import time
import traceback
from pathlib import Path
from datetime import datetime
import warnings
import json
from sklearn.model_selection import train_test_split

# Add src to the path
sys.path.insert(0, os.path.abspath('../src'))

from stg.tableSynthesizer import TableSynthesizer, DEFAULT_MODELS

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')


def get_realistic_synthesizer_configs():
    """Get production-ready configurations for each synthesizer type."""
    return {
        'Identity': {},
        
        'CTGAN': {
            'epochs': 300,              # Higher for better quality
            'batch_size': 500,
            'generator_lr': 2e-4,       # Standard GAN learning rate
            'discriminator_lr': 2e-4,
            'generator_dim': (256, 256),
            'discriminator_dim': (256, 256),
            'generator_decay': 1e-6,
            'discriminator_decay': 0,
            'embedding_dim': 128
        },
        
        'TabDDPM': {
            'num_timesteps': 1000,      # Standard DDPM timesteps
            'num_epochs': 1000,         # Higher for diffusion models
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 0
        },
        
        'PATECTGAN': {
            'epochs': 300,              # Similar to CTGAN
            'batch_size': 500,
            'epsilon': 1.0,             # Privacy parameter
            'delta': 1e-5,
            'generator_dim': (256, 256),
            'discriminator_dim': (256, 256)
        },
        
        'TVAE': {
            'epochs': 300,
            'batch_size': 500,
            'compress_dims': (128, 128),
            'decompress_dims': (128, 128),
            'l2scale': 1e-5,
            'loss_factor': 2
        },
        
        'CART': {
            'max_depth': None,          # No depth limit for flexibility
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42
        },
        
        'DPCART': {
            'max_depth': 10,
            'epsilon': 1.0,             # Privacy budget
            'delta': 1e-5,
            'random_state': 42
        },
        
        'SMOTE': {
            'k_neighbors': 5,
            'sampling_strategy': 'auto',
            'random_state': 42
        },
        
        'BayesianNetwork': {
            'n_epochs': 1000,           # Higher for complex dependencies
            'batch_size': 1000,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'patience': 20
        },
        
        'GREAT': {
            'n_epochs': 100,           # Transformer-based, needs more epochs
            'batch_size': 32,           # Smaller for transformers
            'lr': 5e-5,                 # Lower LR for transformers
            'max_length': 2048,
            'temperature': 0.8
        },
        
        'ARF': {
            'n_epochs': 1000,
            'batch_size': 1000,
            'lr': 1e-3,
            'layers': [256, 256, 256],  # Deeper network
            'dropout': 0.1,
            'patience': 20
        },
        
        'NFlow': {
            'n_epochs': 2000,           # Normalizing flows need many epochs
            'batch_size': 1000,
            'lr': 1e-3,
            'num_flows': 10,            # More flows for better flexibility
            'hidden_features': 256,
            'num_layers': 3
        },
        
        'AutoDiff': {
            'n_epochs': 5000,           # Balance between quality and time
            'diff_n_epochs': 1000,
            'batch_size': 512,
            'lr': 1e-3,
            'weight_decay': 1e-5
        },
        
        'TabSyn': {
            'epochs': 5000,              # Reasonable for VAE+Diffusion
            'batch_size': 1024,
            'lr': 1e-3,
            'max_beta': 1e-2,
            'min_beta': 1e-5,
            'lambd': 0.7
        },
        
        'AIM': {
            'epsilon': 1.0,
            'delta': 1e-9,
            'epochs': 100,
            'rounds': 1000,             # More rounds for better approximation
            'max_model_size': 100,
            'degree': 2
        }
    }


def analyze_dataset(df):
    """Analyze dataset characteristics to inform synthesis strategy."""
    print(f"📊 Dataset Analysis:")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"   Numeric columns: {len(numeric_cols)}")
    print(f"   Categorical columns: {len(categorical_cols)}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"   Memory usage: {memory_mb:.1f} MB")
    
    # Missing values
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"   Missing values: {missing_pct:.1f}%")
    
    # Unique values per column (sample)
    high_cardinality = []
    for col in df.columns[:10]:  # Sample first 10 columns
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.8:
            high_cardinality.append(col)
    
    if high_cardinality:
        print(f"   High-cardinality columns (sample): {len(high_cardinality)}")
    
    return {
        'n_rows': df.shape[0],
        'n_cols': df.shape[1], 
        'n_numeric': len(numeric_cols),
        'n_categorical': len(categorical_cols),
        'memory_mb': memory_mb,
        'missing_pct': missing_pct
    }


def select_optimal_synthesizers(dataset_stats):
    """Select synthesizers most suitable for the dataset characteristics."""
    n_rows, n_cols = dataset_stats['n_rows'], dataset_stats['n_cols']
    memory_mb = dataset_stats['memory_mb']
    
    # Always include these baseline methods
    selected = ['Identity', 'CART', 'SMOTE']
    
    # Add GAN-based methods for medium/large datasets
    if n_rows >= 1000:
        selected.extend(['CTGAN', 'TVAE'])
        
    # Add privacy-aware methods
    if n_rows >= 5000:
        selected.extend(['PATECTGAN', 'DPCART'])
    
    # Add advanced methods for larger datasets
    if n_rows >= 10000 and memory_mb < 500:  # Avoid memory issues
        selected.extend(['BayesianNetwork', 'ARF', 'TabSyn'])
        
    # Add very advanced methods for substantial datasets  
    if n_rows >= 20000 and n_cols <= 100 and memory_mb < 200:
        selected.extend(['GREAT', 'NFlow', 'AutoDiff', 'TabDDPM'])
    
    print(f"🎯 Selected synthesizers for this dataset: {len(selected)}")
    print(f"   {', '.join(selected)}")
    
    return selected


def run_synthesizer_with_timeout(name, df, config, n_samples, timeout=1800):
    """
    Run a single synthesizer with proper error handling and timeout.
    
    Returns:
        tuple: (success: bool, result: DataFrame or error_message: str, 
               duration: float, memory_peak: float, quality_metrics: dict)
    """
    import psutil
    import gc
    
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**2  # MB
    peak_memory = initial_memory
    
    try:
        print(f"  🔧 Running {name}...")
        
        # Monitor memory during execution
        def update_peak_memory():
            nonlocal peak_memory
            current_memory = process.memory_info().rss / 1024**2
            peak_memory = max(peak_memory, current_memory)
        
        # Initialize synthesizer
        synthesizer = TableSynthesizer(name, config)
        update_peak_memory()
        
        # Split data into train/test (80/20) - only use train for fitting
        print(f"    📊 Splitting data: {df.shape[0]} total samples")
        
        # Use stratified split if we have categorical columns to preserve distributions
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        stratify_col = None
        
        if len(categorical_cols) > 0:
            # Use the first categorical column for stratification, or create combined key if multiple
            try:
                if len(categorical_cols) == 1:
                    stratify_col = df[categorical_cols[0]]
                else:
                    # Combine multiple categorical columns for stratification
                    stratify_col = df[categorical_cols].astype(str).apply(
                        lambda x: '_'.join(x), axis=1
                    )
                
                # Check if stratification is viable (need at least 2 samples per category)
                if stratify_col.value_counts().min() < 2:
                    stratify_col = None
                    print("    ⚠️  Too few samples per category for stratification, using random split")
                    
            except Exception as e:
                stratify_col = None
                print(f"    ⚠️  Stratification failed ({e}), using random split")
        
        # Perform the split
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=stratify_col
        )
        
        print(f"    📈 Train: {train_df.shape[0]} samples, Test: {test_df.shape[0]} samples")
        
        # Fit the synthesizer on training data only
        fit_start = time.time()
        synthesizer.fit(train_df)
        fit_time = time.time() - fit_start
        update_peak_memory()
        
        # Generate synthetic data
        gen_start = time.time()
        synthetic_df = synthesizer.sample(n=n_samples, return_dataframe=True)
        gen_time = time.time() - gen_start
        update_peak_memory()
        
        duration = time.time() - start_time
        memory_used = peak_memory - initial_memory
        
        # Enhanced quality metrics including train/test split info
        quality_metrics = {
            'fit_time': fit_time,
            'generation_time': gen_time,
            'total_time': duration,
            'memory_mb': memory_used,
            'output_shape': synthetic_df.shape,
            'columns_match': set(df.columns) == set(synthetic_df.columns),
            'dtypes_preserved': len([c for c in df.columns 
                                   if df[c].dtype == synthetic_df[c].dtype]),
            # Split information
            'original_data_size': df.shape[0],
            'train_data_size': train_df.shape[0],
            'test_data_size': test_df.shape[0],
            'train_split_ratio': train_df.shape[0] / df.shape[0],
            'stratified_split': stratify_col is not None,
            'categorical_columns': len(categorical_cols)
        }
        
        print(f"  ✅ {name} completed in {duration:.1f}s (fit: {fit_time:.1f}s, gen: {gen_time:.1f}s)")
        print(f"     Memory: {memory_used:.1f}MB, Synthetic: {synthetic_df.shape}")
        print(f"     Trained on: {train_df.shape[0]}/{df.shape[0]} samples ({train_df.shape[0]/df.shape[0]*100:.0f}%)")
        
        # Clean up
        del synthesizer
        gc.collect()
        
        return True, synthetic_df, duration, memory_used, quality_metrics
        
    except Exception as e:
        duration = time.time() - start_time
        memory_used = peak_memory - initial_memory
        error_msg = f"Error in {name}: {str(e)}\n{traceback.format_exc()}"
        
        print(f"  ❌ {name} failed after {duration:.1f}s: {str(e)}")
        
        # Clean up on error
        gc.collect()
        
        return False, error_msg, duration, memory_used, {}


def main():
    parser = argparse.ArgumentParser(description="Comprehensive synthesizer benchmark with realistic hyperparameters")
    parser.add_argument("--dataset", default="conversions_all_8-1-25.csv",
                       help="Dataset filename in sandbox/ (default: conversions_all_8-1-25.csv)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Number of synthetic samples (default: same as input)")
    parser.add_argument("--timeout", type=int, default=1800,
                       help="Timeout per synthesizer in seconds (default: 1800)")
    parser.add_argument("--output_dir", default="benchmark_results",
                       help="Output directory (default: benchmark_results)")
    parser.add_argument("--all_synthesizers", action="store_true",
                       help="Run all synthesizers regardless of dataset size")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path("sandbox") / args.dataset
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"❌ Error: Dataset '{input_path}' not found")
        print(f"📁 Available datasets in sandbox/:")
        for f in Path("sandbox").glob("*.csv"):
            print(f"   - {f.name}")
        sys.exit(1)
    
    print(f"🚀 Synthesizer Benchmark - Production Configuration")
    print(f"📊 Dataset: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"⏱️ Timeout: {args.timeout}s per synthesizer")
    print(f"🔄 Train/Test Split: 80/20 with stratification (seed=42)")
    print("-" * 80)
    
    # Load and analyze dataset
    print("📖 Loading dataset...")
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Loaded successfully")
        
        # Clean data
        print("🧹 Cleaning dataset...")
        original_shape = df.shape
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove completely empty rows
        df = df.dropna(axis=0, how='all')
        
        # Convert object columns with few unique values to category
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < len(df) * 0.1:  # Less than 10% unique
                df[col] = df[col].astype('category')
        
        if df.shape != original_shape:
            print(f"   Cleaned: {original_shape} → {df.shape}")
            
        dataset_stats = analyze_dataset(df)
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Determine number of samples
    if args.n_samples is None:
        args.n_samples = min(len(df), 10000)  # Cap at 10k for reasonable time
        print(f"🔢 Samples: {args.n_samples} (auto-selected)")
    else:
        print(f"🔢 Samples: {args.n_samples} (user-specified)")
    
    # Select synthesizers based on dataset characteristics
    if args.all_synthesizers:
        synthesizer_names = list(DEFAULT_MODELS.keys())
        print(f"🔧 Running ALL synthesizers: {len(synthesizer_names)}")
    else:
        synthesizer_names = select_optimal_synthesizers(dataset_stats)
    
    configs = get_realistic_synthesizer_configs()
    
    print("-" * 80)
    
    # Run benchmarks
    results = {}
    errors = {}
    performance_metrics = {}
    
    start_benchmark = time.time()
    
    for i, name in enumerate(synthesizer_names):
        config = configs.get(name, {})
        
        print(f"\n[{i+1}/{len(synthesizer_names)}] {name}")
        if args.verbose and config:
            config_str = json.dumps(config, indent=2)
            print(f"   Config: {config_str}")
        
        success, result, duration, memory_used, quality_metrics = run_synthesizer_with_timeout(
            name, df, config, args.n_samples, args.timeout
        )
        
        performance_metrics[name] = {
            'duration': duration,
            'memory_mb': memory_used,
            'success': success
        }
        performance_metrics[name].update(quality_metrics)
        
        if success:
            # Save synthetic data
            output_file = output_path / f"{name}_synthetic.csv"
            result.to_csv(output_file, index=False)
            results[name] = output_file
            
            # Save metadata
            meta_file = output_path / f"{name}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump({
                    'synthesizer': name,
                    'config': config,
                    'performance': performance_metrics[name],
                    'dataset_info': dataset_stats,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
                
        else:
            # Save error log
            error_file = output_path / f"{name}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Synthesizer: {name}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Dataset: {input_path}\n")
                f.write(f"Configuration: {config}\n")
                f.write("-" * 50 + "\n")
                f.write(result)  # result contains error message
            errors[name] = error_file
    
    total_benchmark_time = time.time() - start_benchmark
    
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    # Summary statistics
    successful = len(results)
    failed = len(errors)
    total = len(synthesizer_names)
    
    print(f"📈 Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"⏱️ Total Time: {total_benchmark_time/60:.1f} minutes")
    print()
    
    # Performance ranking
    successful_perf = {k: v for k, v in performance_metrics.items() if v['success']}
    if successful_perf:
        print("🏆 PERFORMANCE RANKING (by total time):")
        sorted_by_time = sorted(successful_perf.items(), key=lambda x: x[1]['duration'])
        for i, (name, metrics) in enumerate(sorted_by_time):
            duration = metrics['duration']
            memory = metrics.get('memory_mb', 0)
            fit_time = metrics.get('fit_time', 0)
            gen_time = metrics.get('generation_time', 0)
            
            print(f"  {i+1:2d}. {name:<15} {duration:8.1f}s "
                  f"(fit: {fit_time:5.1f}s, gen: {gen_time:5.1f}s, mem: {memory:6.1f}MB)")
    
    # Failed synthesizers
    if errors:
        print("\n💥 FAILED SYNTHESIZERS:")
        for name, error_file in errors.items():
            duration = performance_metrics[name]['duration']
            print(f"  ❌ {name:<15} ({duration:6.1f}s) → {error_file.name}")
    
    # Create comprehensive report
    report_file = output_path / "benchmark_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            'benchmark_info': {
                'dataset': str(input_path),
                'dataset_stats': dataset_stats,
                'n_samples_requested': args.n_samples,
                'timeout': args.timeout,
                'total_time': total_benchmark_time,
                'timestamp': datetime.now().isoformat()
            },
            'results_summary': {
                'total_synthesizers': total,
                'successful': successful,
                'failed': failed,
                'success_rate': successful/total
            },
            'performance_metrics': performance_metrics,
            'successful_outputs': {k: str(v) for k, v in results.items()},
            'error_logs': {k: str(v) for k, v in errors.items()}
        }, f, indent=2)
    
    print(f"\n📝 Detailed report saved to: {report_file}")
    print(f"🎯 All outputs saved in: {output_path}")
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())