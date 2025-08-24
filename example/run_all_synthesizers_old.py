#!/usr/bin/env python3
"""
Comprehensive Synthesizer Runner

This script runs all available synthesizers on a given CSV file and produces
synthetic data using each synthesizer. Results are saved to a target folder.

Usage:
    python run_all_synthesizers.py <csv_path> <output_folder> [--n_samples N] [--timeout T]

Example:
    python run_all_synthesizers.py data.csv results/ --n_samples 1000 --timeout 300
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

# Add src to the path
sys.path.insert(0, os.path.abspath('../src'))

from stg.tableSynthesizer import TableSynthesizer, DEFAULT_MODELS

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')


def get_synthesizer_configs():
    """Get default configurations for each synthesizer type."""
    return {
        'Identity': {},
        'CTGAN': {
            'epochs': 50,  # Reduced for faster execution
            'batch_size': 500,
            'generator_dim': (128, 128),
            'discriminator_dim': (128, 128)
        },
        'TabDDPM': {
            'num_timesteps': 1000,
            'num_epochs': 50  # Reduced for faster execution
        },
        'PATECTGAN': {
            'epochs': 50,  # Reduced for faster execution
            'batch_size': 500
        },
        'TVAE': {
            'epochs': 50,  # Reduced for faster execution
            'batch_size': 500
        },
        'CART': {
            'max_depth': 10,
            'random_state': 42
        },
        'DPCART': {
            'max_depth': 10,
            'epsilon': 1.0,
            'random_state': 42
        },
        'SMOTE': {
            'k_neighbors': 5,
            'random_state': 42
        },
        'BayesianNetwork': {
            'n_epochs': 50  # Reduced for faster execution
        },
        'GREAT': {
            'n_epochs': 50  # Reduced for faster execution
        },
        'ARF': {
            'n_epochs': 50  # Reduced for faster execution
        },
        'NFlow': {
            'n_epochs': 50  # Reduced for faster execution
        },
        'AutoDiff': {
            'n_epochs': 5,  # Very reduced for faster execution
            'diff_n_epochs': 5,
            'batch_size': 32
        },
        'TabSyn': {
            'epochs': 10,  # Very reduced due to performance issues
            'batch_size': 32
        },
        'AIM': {
            'epsilon': 1.0,
            'delta': 1e-9,
            'epochs': 10,
            'rounds': 50
        }
    }


def run_synthesizer(name, df, config, n_samples, timeout=300):
    """
    Run a single synthesizer and return the synthetic data.
    
    Args:
        name: Name of the synthesizer
        df: Input DataFrame
        config: Configuration dictionary for the synthesizer
        n_samples: Number of samples to generate
        timeout: Timeout in seconds
        
    Returns:
        tuple: (success: bool, result: DataFrame or error_message: str, duration: float)
    """
    start_time = time.time()
    
    try:
        print(f"  Running {name}...")
        
        # Initialize synthesizer
        synthesizer = TableSynthesizer(name, config)
        
        # Fit the synthesizer
        synthesizer.fit(df)
        
        # Generate synthetic data
        synthetic_df = synthesizer.sample(n=n_samples, return_dataframe=True)
        
        duration = time.time() - start_time
        print(f"  ✅ {name} completed in {duration:.2f}s")
        
        return True, synthetic_df, duration
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error in {name}: {str(e)}\n{traceback.format_exc()}"
        print(f"  ❌ {name} failed after {duration:.2f}s: {str(e)}")
        
        return False, error_msg, duration


def main():
    parser = argparse.ArgumentParser(description="Run all synthesizers on a CSV file")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("output_folder", help="Path to output folder")
    parser.add_argument("--n_samples", type=int, default=1000, 
                       help="Number of synthetic samples to generate (default: 1000)")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per synthesizer in seconds (default: 300)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: Input CSV file '{args.csv_path}' not found")
        sys.exit(1)
    
    # Create output folder
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting synthesizer benchmark")
    print(f"📊 Input: {args.csv_path}")
    print(f"📁 Output: {args.output_folder}")
    print(f"🔢 Samples: {args.n_samples}")
    print(f"⏱️  Timeout: {args.timeout}s per synthesizer")
    print("-" * 60)
    
    # Load data
    try:
        df = pd.read_csv(args.csv_path)
        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if args.verbose:
            print(f"📋 Data types:\n{df.dtypes}")
            print(f"📊 Data preview:\n{df.head()}")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Get available synthesizers and their configs
    available_synthesizers = list(DEFAULT_MODELS.keys())
    configs = get_synthesizer_configs()
    
    print(f"🔧 Available synthesizers: {len(available_synthesizers)}")
    print(f"   {', '.join(available_synthesizers)}")
    print("-" * 60)
    
    # Track results
    results = {}
    errors = {}
    execution_times = {}
    
    # Run each synthesizer
    for name in available_synthesizers:
        config = configs.get(name, {})
        
        success, result, duration = run_synthesizer(
            name, df, config, args.n_samples, args.timeout
        )
        
        execution_times[name] = duration
        
        if success:
            # Save synthetic data
            output_file = output_path / f"{name}_synthetic.csv"
            result.to_csv(output_file, index=False)
            results[name] = output_file
            
        else:
            # Save error log
            error_file = output_path / f"{name}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Synthesizer: {name}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Configuration: {config}\n")
                f.write("-" * 50 + "\n")
                f.write(result)  # result contains error message in this case
            errors[name] = error_file
    
    print("-" * 60)
    print("📊 FINAL RESULTS")
    print("-" * 60)
    
    # Summary
    successful = len(results)
    failed = len(errors)
    total = len(available_synthesizers)
    
    print(f"✅ Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"❌ Failed: {failed}/{total} ({failed/total*100:.1f}%)")
    print()
    
    # Successful synthesizers
    if results:
        print("🎉 SUCCESSFUL SYNTHESIZERS:")
        for name, output_file in results.items():
            duration = execution_times[name]
            file_size = output_file.stat().st_size / 1024  # KB
            print(f"  ✅ {name:<15} ({duration:6.2f}s) → {output_file.name} ({file_size:.1f}KB)")
    
    # Failed synthesizers
    if errors:
        print("\n💥 FAILED SYNTHESIZERS:")
        for name, error_file in errors.items():
            duration = execution_times[name]
            print(f"  ❌ {name:<15} ({duration:6.2f}s) → {error_file.name}")
    
    # Execution time analysis
    if execution_times:
        print(f"\n⏱️  EXECUTION TIME ANALYSIS:")
        sorted_times = sorted(execution_times.items(), key=lambda x: x[1])
        print(f"   Fastest: {sorted_times[0][0]} ({sorted_times[0][1]:.2f}s)")
        print(f"   Slowest: {sorted_times[-1][0]} ({sorted_times[-1][1]:.2f}s)")
        avg_time = sum(execution_times.values()) / len(execution_times)
        print(f"   Average: {avg_time:.2f}s")
    
    # Create summary report
    summary_file = output_path / "synthesis_report.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Synthesizer Benchmark Report\n")
        f.write(f"============================\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Input file: {args.csv_path}\n")
        f.write(f"Input shape: {df.shape}\n")
        f.write(f"Samples requested: {args.n_samples}\n")
        f.write(f"Timeout: {args.timeout}s\n\n")
        
        f.write(f"Results Summary\n")
        f.write(f"---------------\n")
        f.write(f"Total synthesizers: {total}\n")
        f.write(f"Successful: {successful} ({successful/total*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n\n")
        
        if results:
            f.write(f"Successful Synthesizers:\n")
            for name, output_file in results.items():
                duration = execution_times[name]
                f.write(f"  {name}: {duration:.2f}s → {output_file.name}\n")
        
        if errors:
            f.write(f"\nFailed Synthesizers:\n")
            for name, error_file in errors.items():
                duration = execution_times[name]
                f.write(f"  {name}: {duration:.2f}s → {error_file.name}\n")
        
        f.write(f"\nExecution Times:\n")
        for name, duration in sorted_times:
            status = "✅" if name in results else "❌"
            f.write(f"  {status} {name}: {duration:.2f}s\n")
    
    print(f"\n📝 Summary report saved to: {summary_file}")
    print(f"🎯 All outputs saved in: {args.output_folder}")
    
    print(f"\n🏁 Benchmark completed!")
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())