import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer

# Check if ARF is available
try:
    from stg.tableSynthesizer import DEFAULT_MODELS
    ARF_AVAILABLE = 'ARF' in DEFAULT_MODELS
except ImportError:
    ARF_AVAILABLE = False


def create_test_dataframe():
    """Create a test DataFrame"""
    np.random.seed(42)  # For reproducible tests
    
    df = pd.DataFrame({
        'numeric_col1': np.random.randn(50),
        'numeric_col2': np.random.uniform(0, 1, 50),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 50),
        'binary_col': np.random.choice([0, 1], 50)
    })
    
    return df


@pytest.mark.skipif(not ARF_AVAILABLE, reason="ARF not available due to missing dependencies")
def test_ARF_initialization():
    """Test ARF model initialization"""
    config = {}
    synthesizer = TableSynthesizer('ARF', config)
    assert synthesizer.model is not None
    print("ARF model initialized successfully")


@pytest.mark.skipif(not ARF_AVAILABLE, reason="ARF not available due to missing dependencies")
def test_ARF_dataframe_support():
    """Test ARF with DataFrame input"""
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Configuration for ARF
    config = {}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer('ARF', config)
    
    # Test fitting with DataFrame (ARF only supports DataFrame)
    synthesizer.fit(df)
    
    # Test tensor output
    n_samples = 20
    sampled_tensor = synthesizer.sample(n=n_samples)
    assert sampled_tensor.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_tensor.shape[0]}"
    assert sampled_tensor.shape[1] == df.shape[1], f"Expected {df.shape[1]} features, got {sampled_tensor.shape[1]}"
    
    # Test DataFrame output
    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    assert sampled_df.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_df.shape[0]}"
    assert isinstance(sampled_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(sampled_df)}"
    
    # Check column names match
    expected_cols = set(df.columns)
    actual_cols = set(sampled_df.columns)
    assert expected_cols == actual_cols, f"Column names mismatch. Expected: {expected_cols}, Got: {actual_cols}"
    
    print("ARF DataFrame test passed successfully!")


@pytest.mark.skipif(not ARF_AVAILABLE, reason="ARF not available due to missing dependencies")
def test_ARF_different_data_types():
    """Test ARF with different data types"""
    np.random.seed(42)
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 40),
        'float_col': np.random.randn(40),
        'category_col': np.random.choice(['X', 'Y', 'Z'], 40),
        'binary_col': np.random.choice(['Yes', 'No'], 40)
    })
    
    config = {}
    synthesizer = TableSynthesizer('ARF', config)
    synthesizer.fit(df)
    
    synthetic_df = synthesizer.sample(n=15, return_dataframe=True)
    
    # Verify output
    assert synthetic_df.shape[0] == 15
    assert synthetic_df.shape[1] == 4
    assert all(col in synthetic_df.columns for col in df.columns)
    
    print("ARF different data types test passed!")


def test_ARF_availability():
    """Test if ARF is properly detected as available or not"""
    if ARF_AVAILABLE:
        print("ARF is available and registered in DEFAULT_MODELS")
    else:
        print("ARF is not available due to missing synthcity dependencies (expected)")
        pytest.skip("ARF not available due to missing dependencies")


if __name__ == "__main__":
    test_ARF_availability()
    if ARF_AVAILABLE:
        test_ARF_initialization()
        test_ARF_dataframe_support() 
        test_ARF_different_data_types()