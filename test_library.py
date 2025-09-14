#!/usr/bin/env python3
"""
Test script to verify the STG library functionality for end users.

This script tests the library as it would be used by external users
after pip installation, without access to internal test utilities.
"""
import pandas as pd
import numpy as np
import torch
import sys, os
# Ensure local src/ is importable for test execution without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from stg import TableSynthesizer

def test_basic_import():
    """Test that the library can be imported correctly."""
    print("✓ STG library imported successfully")
    
    # Check available synthesizers
    from stg.tableSynthesizer import DEFAULT_MODELS
    available_models = list(DEFAULT_MODELS.keys())
    print(f"✓ Available synthesizers: {available_models}")
    
    return available_models

def create_sample_data():
    """Create sample datasets for testing."""
    np.random.seed(42)
    
    # Simple mixed dataset
    simple_df = pd.DataFrame({
        'numeric_feature': np.random.randn(100),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
        'binary_feature': np.random.choice([0, 1], 100),
        'continuous_bounded': np.random.uniform(0, 1, 100)
    })
    
    # Complex dataset with more features
    complex_df = pd.DataFrame({
        'age': np.random.randint(18, 80, 200),
        'income': np.random.lognormal(10, 1, 200),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 200),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 200),
        'has_car': np.random.choice([True, False], 200),
        'satisfaction': np.random.randint(1, 6, 200)
    })
    
    return simple_df, complex_df

def test_synthesizer(model_name, df, config=None):
    """Test a specific synthesizer on given data."""
    print(f"\nTesting {model_name}...")
    
    try:
        # Initialize synthesizer
        synthesizer = TableSynthesizer(model_name, config or {})
        
        # Fit the model
        print(f"  ✓ Fitting {model_name} on data with shape {df.shape}")
        synthesizer.fit(df, batch_size=32)
        
        # Generate samples as tensor
        tensor_samples = synthesizer.sample(n=10)
        assert isinstance(tensor_samples, torch.Tensor), f"Expected torch.Tensor, got {type(tensor_samples)}"
        assert tensor_samples.shape[0] == 10, f"Expected 10 samples, got {tensor_samples.shape[0]}"
        print(f"  ✓ Generated tensor samples: {tensor_samples.shape}")
        
        # Generate samples as DataFrame
        df_samples = synthesizer.sample(n=10, return_dataframe=True)
        assert isinstance(df_samples, pd.DataFrame), f"Expected pd.DataFrame, got {type(df_samples)}"
        assert df_samples.shape[0] == 10, f"Expected 10 samples, got {df_samples.shape[0]}"
        assert set(df_samples.columns) == set(df.columns), "Column mismatch in generated DataFrame"
        print(f"  ✓ Generated DataFrame samples: {df_samples.shape}")
        print(f"  ✓ Column preservation: {list(df_samples.columns)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ {model_name} failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive tests of the STG library."""
    print("=" * 60)
    print("STG LIBRARY FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test 1: Basic import
    available_models = test_basic_import()
    
    # Test 2: Create test data
    print("\n" + "-" * 40)
    print("Creating test datasets...")
    simple_df, complex_df = create_sample_data()
    print(f"✓ Simple dataset: {simple_df.shape}, columns: {simple_df.columns.tolist()}")
    print(f"✓ Complex dataset: {complex_df.shape}, columns: {complex_df.columns.tolist()}")
    
    # Test 3: Test core synthesizers
    print("\n" + "-" * 40)
    print("Testing core synthesizers...")
    
    # Test configurations for different models
    test_configs = {
        'Identity': {},
        'CTGAN': {'epochs': 5, 'batch_size': 32},
        'TVAE': {'epochs': 5, 'batch_size': 32, 'embedding_dim': 64},
        'PATECTGAN': {'epochs': 5, 'batch_size': 32},
        'TabDDPM': {'epochs': 5, 'num_timesteps': 10, 'batch_size': 32},
        'SMOTE': {'k_neighbors': 3, 'random_state': 42},
        'CART': {'max_depth': 5, 'random_state': 42},
        'DPCART': {'max_depth': 5, 'epsilon': 1.0},
        'AutoDiff': {'n_epochs': 5, 'diff_n_epochs': 5, 'batch_size': 32},
        'TabSyn': {'epochs': 5, 'batch_size': 32}
    }
    
    results = {}
    
    # Test each available synthesizer
    for model_name in available_models:
        if model_name in test_configs:
            config = test_configs[model_name]
            # Use simple dataset for faster testing
            success = test_synthesizer(model_name, simple_df, config)
            results[model_name] = success
    
    # Test 4: Advanced usage example
    print("\n" + "-" * 40)
    print("Testing advanced usage...")
    
    try:
        # Multi-step workflow example
        synthesizer = TableSynthesizer('Identity', {})
        synthesizer.fit(complex_df)
        
        # Generate different amounts of data
        small_sample = synthesizer.sample(n=50, return_dataframe=True)
        large_sample = synthesizer.sample(n=500, return_dataframe=True)
        
        print(f"  ✓ Generated small sample: {small_sample.shape}")
        print(f"  ✓ Generated large sample: {large_sample.shape}")
        
        # Verify data types are preserved
        for col in complex_df.columns:
            original_type = complex_df[col].dtype
            sample_type = small_sample[col].dtype
            print(f"  ✓ Column '{col}': {original_type} → {sample_type}")
            
    except Exception as e:
        print(f"  ❌ Advanced usage test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    success_rate = passed / total * 100 if total > 0 else 0
    
    print(f"Overall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    print("\nDetailed Results:")
    for model_name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {model_name:15s}: {status}")
    
    if success_rate >= 80:
        print("\n🎉 Library test PASSED! STG is ready for production use.")
        return True
    else:
        print("\n⚠️  Library test had issues. Some synthesizers may need attention.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
