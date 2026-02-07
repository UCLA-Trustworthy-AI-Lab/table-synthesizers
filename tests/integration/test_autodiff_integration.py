import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer
from utils import run_sandbox_dataset_test

# Check if AutoDiff is actually available (dependencies installed)
try:
    from stg.AutoDiff.autodiff_synthesizer import AUTODIFF_AVAILABLE
except ImportError:
    AUTODIFF_AVAILABLE = False


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


@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff not available due to missing dependencies")
def test_AutoDiff_initialization():
    """Test AutoDiff model initialization"""
    config = {"n_epochs": 10, "diff_n_epochs": 10}  # Reduced epochs for testing
    synthesizer = TableSynthesizer('AutoDiff', config)
    assert synthesizer.model is not None
    print("AutoDiff model initialized successfully")


@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff not available due to missing dependencies")
def test_AutoDiff_dataframe_support():
    """Test AutoDiff with DataFrame input"""
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Configuration for AutoDiff (reduced epochs for testing)
    config = {"n_epochs": 5, "diff_n_epochs": 5, "batch_size": 16}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer('AutoDiff', config)
    
    # Test fitting with DataFrame (AutoDiff only supports DataFrame)
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
    
    print("AutoDiff DataFrame test passed successfully!")


@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff not available due to missing dependencies")
def test_AutoDiff_different_data_types():
    """Test AutoDiff with different data types"""
    np.random.seed(42)
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 40),
        'float_col': np.random.randn(40),
        'category_col': np.random.choice(['X', 'Y', 'Z'], 40),
        'binary_col': np.random.choice(['Yes', 'No'], 40)
    })
    
    config = {"n_epochs": 5, "diff_n_epochs": 5, "batch_size": 16}
    synthesizer = TableSynthesizer('AutoDiff', config)
    synthesizer.fit(df)
    
    synthetic_df = synthesizer.sample(n=15, return_dataframe=True)
    
    # Verify output
    assert synthetic_df.shape[0] == 15
    assert synthetic_df.shape[1] == 4
    assert all(col in synthetic_df.columns for col in df.columns)
    
    print("AutoDiff different data types test passed!")


@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff not available due to missing dependencies")
def test_AutoDiff_sandbox_insurance():
    """Test AutoDiff on insurance dataset"""
    config = {"n_epochs": 10, "diff_n_epochs": 10, "batch_size": 16}
    run_sandbox_dataset_test('AutoDiff', 'insurance', config=config, n_samples=50, sample_ratio=0.1)


@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff not available due to missing dependencies")
def test_AutoDiff_sandbox_titanic():
    """Test AutoDiff on Titanic dataset"""
    config = {"n_epochs": 10, "diff_n_epochs": 10, "batch_size": 16}
    run_sandbox_dataset_test('AutoDiff', 'Titanic', config=config, n_samples=50, sample_ratio=0.2)


def test_AutoDiff_availability():
    """Test if AutoDiff is properly detected as available or not"""
    if AUTODIFF_AVAILABLE:
        print("AutoDiff is available and registered in DEFAULT_MODELS")
    else:
        print("AutoDiff is not available due to missing dependencies (expected)")
        pytest.skip("AutoDiff not available due to missing dependencies")


if __name__ == "__main__":
    test_AutoDiff_availability()
    if AUTODIFF_AVAILABLE:
        test_AutoDiff_initialization()
        test_AutoDiff_dataframe_support() 
        test_AutoDiff_different_data_types()
        
        # Test sandbox datasets
        test_AutoDiff_sandbox_insurance()
        test_AutoDiff_sandbox_titanic()