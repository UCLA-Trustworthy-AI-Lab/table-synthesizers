#!/usr/bin/env python3
"""
CUDA-Optimized Functional Test Suite for main.py with all 16 synthesizers

This script is optimized for NVIDIA CUDA-compatible devices to accelerate
model training using GPU. It includes CUDA availability checks and optimized
batch sizes for GPU memory.
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
    
    # Advanced Deep Learning Models (GPU-accelerated)
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

# Models that benefit most from GPU acceleration
GPU_ACCELERATED_MODELS = ['CTGAN', 'PATECTGAN', 'TabDDPM', 'AutoDiff', 'LTM_VAE', 'TVAE', 'TabSyn', 'GREAT']

# Models that require more time (reduced on GPU)
SLOW_MODELS = ['CTGAN', 'PATECTGAN', 'TabDDPM', 'AutoDiff', 'LTM_VAE', 'GREAT']

# Model-specific timeouts for CUDA (reduced from CPU timeouts)
# GPU training is typically 3-10x faster than CPU
MODEL_TIMEOUTS_CUDA = {
    'AutoDiff': 120,      # VAE + Diffusion hybrid (reduced from 300s)
    'CTGAN': 90,          # GAN training (reduced from 240s)
    'PATECTGAN': 90,      # Privacy-aware GAN (reduced from 240s)
    'LTM_VAE': 60,        # Large latent model (reduced from 180s)
    'GREAT': 60,          # Transformer-based (reduced from 180s)
    'TabDDPM': 60,        # Diffusion model (reduced from 180s)
    'TabSyn': 120,        # Subprocess-based (reduced from 3000s)
    'TVAE': 30,           # Fast on GPU
}

# Fallback CPU timeouts (same as original)
MODEL_TIMEOUTS_CPU = {
    'AutoDiff': 300,
    'CTGAN': 240,
    'PATECTGAN': 240,
    'LTM_VAE': 180,
    'GREAT': 180,
    'TabDDPM': 180,
    'TabSyn': 3000,
}

def check_cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return True, device_count, device_name
        return False, 0, None
    except ImportError:
        return False, 0, None

def run_model_test(model_name, config_path, data_path, output_path, timeout=120, use_cuda=False):
    """
    Test a single model using main.py
    
    Args:
        model_name: Name of the model to test
        config_path: Path to config file
        data_path: Path to transformed data
        output_path: Path to output directory
        timeout: Maximum time to wait (seconds)
        use_cuda: Whether CUDA is available
    
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
    
    # Set CUDA environment variable if available
    env = os.environ.copy()
    if use_cuda:
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    if use_cuda and model_name in GPU_ACCELERATED_MODELS:
        print(f"🚀 GPU-Accelerated")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
            env=env
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
            'gpu_accelerated': use_cuda and model_name in GPU_ACCELERATED_MODELS,
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
            'gpu_accelerated': use_cuda and model_name in GPU_ACCELERATED_MODELS,
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
            'gpu_accelerated': use_cuda and model_name in GPU_ACCELERATED_MODELS,
            'stdout': '',
            'stderr': ''
        }

def print_result(result):
    """Print a single test result"""
    gpu_indicator = " 🚀" if result.get('gpu_accelerated', False) else ""
    print(f"{result['status']} {result['model']:<20} ({result['time']:.2f}s){gpu_indicator}")
    if result['error']:
        print(f"   Error: {result['error']}")

def print_summary(results, use_cuda):
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
    
    if use_cuda:
        gpu_accelerated = [r for r in passed if r.get('gpu_accelerated', False)]
        print(f"🚀 GPU-Accelerated: {len(gpu_accelerated)}")
    
    success_rate = (len(passed) / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    # Calculate average speedup for GPU models
    if use_cuda and passed:
        gpu_times = [r['time'] for r in passed if r.get('gpu_accelerated', False)]
        if gpu_times:
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            print(f"Avg GPU model time: {avg_gpu_time:.2f}s")
    
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
    
    parser = argparse.ArgumentParser(description='Test all models with main.py (CUDA-optimized)')
    parser.add_argument('--mode', choices=['all', 'fast', 'slow', 'gpu'], default='fast',
                       help='Which models to test (default: fast, gpu=GPU-accelerated models only)')
    parser.add_argument('--config-path', default='example_run/config.json',
                       help='Path to config file')
    parser.add_argument('--transformed-data-path', default='example_run/transformed_data',
                       help='Path to transformed data')
    parser.add_argument('--synthesizer-output-path', default='example_run/output',
                       help='Path to output directory')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Default timeout per model in seconds (default: 120)')
    parser.add_argument('--models', nargs='+', 
                       help='Specific models to test (overrides --mode)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU mode even if CUDA is available')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    cuda_available, device_count, device_name = check_cuda_available()
    use_cuda = cuda_available and not args.force_cpu
    
    # Select appropriate timeouts
    MODEL_TIMEOUTS = MODEL_TIMEOUTS_CUDA if use_cuda else MODEL_TIMEOUTS_CPU
    
    # Determine which models to test
    if args.models:
        models_to_test = args.models
    elif args.mode == 'all':
        models_to_test = ALL_MODELS
    elif args.mode == 'fast':
        models_to_test = FAST_MODELS
    elif args.mode == 'gpu':
        models_to_test = GPU_ACCELERATED_MODELS
    else:  # slow
        models_to_test = SLOW_MODELS
    
    print(f"\n{'='*80}")
    print(f"CUDA-OPTIMIZED FUNCTIONAL TEST SUITE FOR main.py")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Models to test: {len(models_to_test)}")
    
    if use_cuda:
        print(f"🚀 CUDA: ENABLED")
        print(f"   Devices: {device_count}")
        print(f"   GPU: {device_name}")
        print(f"   Timeouts: Optimized for GPU (3-10x faster)")
    else:
        if args.force_cpu:
            print(f"⚠️  CUDA: DISABLED (forced CPU mode)")
        else:
            print(f"⚠️  CUDA: NOT AVAILABLE (using CPU)")
        print(f"   Timeouts: Standard CPU timeouts")
    
    print(f"Default timeout: {args.timeout}s")
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
            timeout=model_timeout,
            use_cuda=use_cuda
        )
        results.append(result)
        print_result(result)
    
    total_time = time.time() - total_start
    
    # Print summary
    print_summary(results, use_cuda)
    
    print(f"\nTotal execution time: {total_time:.2f}s")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    all_passed = all('✅' in r['status'] for r in results)
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
