#!/usr/bin/env python3
"""
FOCUSED TEST FOR 4 FAILING MODELS
Test and fix specific issues with SMOTE, AutoDiff, TabSyn, and LTM_VAE
"""

import sys
import os
import pandas as pd
import numpy as np
import time

# Add src/stg to path to find zero_workaround
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'stg'))

def apply_zero_workaround():
    """Apply the zero workaround"""
    import zero_workaround as zero
    sys.modules['zero'] = zero
    return zero

def test_single_model(model_name, test_data):
    """Test a single model with detailed error reporting"""
    print(f"\n{'='*50}")
    print(f"🔍 TESTING {model_name}")
    print(f"{'='*50}")

    start_time = time.time()

    try:
        from stg.tableSynthesizer import TableSynthesizer

        # Specific configs for problematic models
        configs = {
            'SMOTE': {'k_neighbors': 3},  # Lower k_neighbors for small dataset
            'AutoDiff': {'epochs': 1},
            'TabSyn': {'epochs': 1, 'batch_size': 8},
            'LTM_VAE': {'model_type': 'vae', 'config_task': 'quick_test'}
        }

        config = configs.get(model_name, {})
        print(f"📦 Creating {model_name} with config: {config}")

        # Try to create model
        model = TableSynthesizer(model_name, config)
        print("✅ Model created successfully")

        # Check interface
        has_fit = hasattr(model.model, 'fit')
        has_sample = hasattr(model.model, 'sample')
        print(f"🔧 Interface: fit={has_fit}, sample={has_sample}")

        if not (has_fit and has_sample):
            return {
                'model_name': model_name,
                'status': 'INTERFACE_MISSING',
                'error': f'Missing fit={has_fit} or sample={has_sample}',
                'time': time.time() - start_time
            }

        # Try fitting
        print(f"🏋️ Training {model_name}...")
        fit_start = time.time()
        model.model.fit(test_data)
        fit_time = time.time() - fit_start
        print(f"✅ Training completed in {fit_time:.2f}s")

        # Try sampling
        print(f"🎲 Testing sampling...")
        sample_start = time.time()
        samples = model.model.sample(3, return_dataframe=True)
        sample_time = time.time() - sample_start
        print(f"✅ Sampling completed in {sample_time:.3f}s")

        # Validate output
        print(f"🔍 Output type: {type(samples)}")
        print(f"🔍 Output length: {len(samples) if hasattr(samples, '__len__') else 'N/A'}")

        if isinstance(samples, pd.DataFrame):
            print(f"✅ Output validation: DataFrame with {samples.shape}")
            print(f"   Columns match: {set(samples.columns) == set(test_data.columns)}")
            print(f"   Sample data preview:")
            print(samples.head(2))

            total_time = time.time() - start_time
            return {
                'model_name': model_name,
                'status': 'SUCCESS',
                'fit_time': fit_time,
                'sample_time': sample_time,
                'total_time': total_time
            }
        else:
            return {
                'model_name': model_name,
                'status': 'OUTPUT_ISSUE',
                'error': f'Expected DataFrame, got {type(samples)} with length {len(samples) if hasattr(samples, "__len__") else "N/A"}',
                'time': time.time() - start_time
            }

    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ {model_name} FAILED:")
        print(f"   Error: {str(e)}")

        # Print full traceback for debugging
        import traceback
        print(f"   Full traceback:")
        traceback.print_exc()

        return {
            'model_name': model_name,
            'status': 'FAILED',
            'error': str(e),
            'time': error_time
        }

def main():
    print("🔧 FOCUSED TEST FOR 4 FAILING MODELS")
    print("=" * 60)

    # Apply workaround
    zero = apply_zero_workaround()
    print("✅ Zero workaround applied")

    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'num': np.random.normal(0, 1, 50),  # Larger dataset for SMOTE
        'cat': np.random.choice(['A', 'B', 'C'], 50),
        'bin': np.random.choice([0, 1], 50)
    })
    print(f"📊 Test data: {test_data.shape}")

    # Test the 4 failing models
    failing_models = ['SMOTE', 'AutoDiff', 'TabSyn', 'LTM_VAE']

    results = []
    for i, model_name in enumerate(failing_models, 1):
        print(f"\n{'#'*60}")
        print(f"TEST {i}/4: {model_name}")
        print(f"{'#'*60}")

        result = test_single_model(model_name, test_data)
        results.append(result)

        print(f"\n📊 {model_name} Result: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"   ✅ Total time: {result['total_time']:.2f}s")
        else:
            print(f"   ❌ Error: {result['error']}")

    # Summary
    print(f"\n{'='*60}")
    print("🎯 FOCUSED TEST RESULTS")
    print(f"{'='*60}")

    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {result['model_name']:<12} | {result['status']}")
        if result['status'] != 'SUCCESS':
            print(f"     └─ {result['error'][:60]}...")

    success_count = len([r for r in results if r['status'] == 'SUCCESS'])
    print(f"\n📈 Fixed: {success_count}/4 models")

if __name__ == "__main__":
    main()