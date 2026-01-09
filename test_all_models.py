#!/usr/bin/env python3
"""
Functional Test Suite for main.py with all 16 synthesizers

This script tests each model using the --select-training-model parameter
to ensure all models work correctly with main.py.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

# All 16 available models
ALL_MODELS = [
    # Core Models (Fast & Production-Ready)
    'Identity',
    'TVAE', 
    'SMOTE',
    'CART',
    'DPCART',
    'AIM',
    
    # Advanced Deep Learning Models
    'TabDDPM',
    'AutoDiff',
    'LTM_VAE',
    'TabSyn',
    'CTGAN',
    'PATECTGAN',
    
    # Synthcity-Based Models
    'ARF',
    'NFlow',
    'BayesianNetwork',
    'GREAT'
]

# Models known to be fast (for quick testing)
FAST_MODELS = ['Identity', 'TVAE', 'SMOTE', 'CART', 'DPCART', 'AIM', 'ARF']

# Models that require more time
SLOW_MODELS = ['CTGAN', 'PATECTGAN', 'TabDDPM', 'AutoDiff', 'LTM_VAE', 'GREAT']

# Model-specific timeouts (in seconds)
# Models not listed here will use the default timeout
MODEL_TIMEOUTS = {
    'AutoDiff': 300,      # VAE + Diffusion hybrid needs more time
    'CTGAN': 240,         # GAN training is slow
    'PATECTGAN': 240,     # Privacy-aware GAN is slow
    'LTM_VAE': 180,       # Large latent model
    'GREAT': 180,         # Transformer-based
    'TabDDPM': 180,       # Diffusion model
    'TabSyn': 3000,        # Subprocess-based synthesis needs more time
}

def run_model_test(model_name, config_path, data_path, output_path, timeout=120):
    """
    Test a single model using main.py
    
    Args:
        model_name: Name of the model to test
        config_path: Path to config file
        data_path: Path to transformed data
        output_path: Path to output directory
        timeout: Maximum time to wait (seconds)
    
    Returns:
        dict: Test results with status, time, and error info
    """
    cmd = [
        'python', 'main.py',
        '--select-training-model', model_name,
        '--config-path', config_path,
        '--transformed-data-path', data_path,
        '--synthesizer-output-path', output_path
    ]
    
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        elapsed_time = time.time() - start_time
        
        # Check if successful
        if result.returncode == 0:
            # Verify output file exists
            output_file = os.path.join(output_path, 'synthetic_data.csv')
            if os.path.exists(output_file):
                # Check file size
                file_size = os.path.getsize(output_file)
                if file_size > 0:
                    status = "✅ PASS"
                    error = None
                else:
                    status = "⚠️  WARN"
                    error = "Output file is empty"
            else:
                status = "❌ FAIL"
                error = "Output file not created"
        else:
            status = "❌ FAIL"
            error = f"Exit code: {result.returncode}"
            
            # Extract error message from stderr
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                # Get last few lines which usually contain the actual error
                relevant_errors = [line for line in error_lines if 'Error' in line or 'Exception' in line]
                if relevant_errors:
                    error += f"\n{relevant_errors[-1]}"
        
        return {
            'model': model_name,
            'status': status,
            'time': elapsed_time,
            'error': error,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        return {
            'model': model_name,
            'status': '⏱️  TIMEOUT',
            'time': elapsed_time,
            'error': f'Exceeded {timeout}s timeout',
            'stdout': '',
            'stderr': ''
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'model': model_name,
            'status': '💥 ERROR',
            'time': elapsed_time,
            'error': str(e),
            'stdout': '',
            'stderr': ''
        }

def print_result(result):
    """Print a single test result"""
    print(f"{result['status']} {result['model']:<20} ({result['time']:.2f}s)")
    if result['error']:
        print(f"   Error: {result['error']}")

def print_summary(results):
    """Print summary of all test results"""
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed = [r for r in results if '✅' in r['status']]
    warned = [r for r in results if '⚠️' in r['status']]
    failed = [r for r in results if '❌' in r['status']]
    timeout = [r for r in results if '⏱️' in r['status']]
    errored = [r for r in results if '💥' in r['status']]
    
    total = len(results)
    
    print(f"Total Tests:    {total}")
    print(f"✅ Passed:      {len(passed)}")
    print(f"⚠️  Warnings:    {len(warned)}")
    print(f"❌ Failed:      {len(failed)}")
    print(f"⏱️  Timeouts:    {len(timeout)}")
    print(f"💥 Errors:      {len(errored)}")
    
    success_rate = (len(passed) / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if failed or timeout or errored:
        print(f"\n{'='*80}")
        print("FAILED/PROBLEMATIC MODELS:")
        print(f"{'='*80}")
        for r in failed + timeout + errored:
            print(f"\n{r['status']} {r['model']}")
            print(f"   Time: {r['time']:.2f}s")
            if r['error']:
                print(f"   Error: {r['error']}")
    
    return success_rate >= 80  # Consider test suite successful if 80%+ pass

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test all models with main.py')
    parser.add_argument('--mode', choices=['all', 'fast', 'slow'], default='fast',
                       help='Which models to test (default: fast)')
    parser.add_argument('--config-path', default='example_run/config.json',
                       help='Path to config file')
    parser.add_argument('--transformed-data-path', default='example_run/transformed_data',
                       help='Path to transformed data')
    parser.add_argument('--synthesizer-output-path', default='example_run/output',
                       help='Path to output directory')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Timeout per model in seconds (default: 120)')
    parser.add_argument('--models', nargs='+', 
                       help='Specific models to test (overrides --mode)')
    
    args = parser.parse_args()
    
    # Determine which models to test
    if args.models:
        models_to_test = args.models
    elif args.mode == 'all':
        models_to_test = ALL_MODELS
    elif args.mode == 'fast':
        models_to_test = FAST_MODELS
    else:  # slow
        models_to_test = SLOW_MODELS
    
    print(f"\n{'='*80}")
    print(f"FUNCTIONAL TEST SUITE FOR main.py")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Models to test: {len(models_to_test)}")
    print(f"Timeout per model: {args.timeout}s")
    print(f"{'='*80}")
    
    # Verify example data exists
    if not os.path.exists(args.config_path):
        print(f"\n❌ ERROR: Config file not found at {args.config_path}")
        print("Please run: python create_example_data.py")
        sys.exit(1)
    
    if not os.path.exists(args.transformed_data_path):
        print(f"\n❌ ERROR: Transformed data not found at {args.transformed_data_path}")
        print("Please run: python create_example_data.py")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.synthesizer_output_path, exist_ok=True)
    
    # Run tests
    results = []
    total_start = time.time()
    
    for i, model in enumerate(models_to_test, 1):
        print(f"\n[{i}/{len(models_to_test)}] ", end='')
        
        # Use model-specific timeout if available, otherwise use default
        model_timeout = MODEL_TIMEOUTS.get(model, args.timeout)
        
        result = run_model_test(
            model,
            args.config_path,
            args.transformed_data_path,
            args.synthesizer_output_path,
            timeout=model_timeout
        )
        results.append(result)
        print_result(result)
    
    total_time = time.time() - total_start
    
    # Print summary
    print_summary(results)
    
    print(f"\nTotal execution time: {total_time:.2f}s")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    all_passed = all('✅' in r['status'] for r in results)
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
