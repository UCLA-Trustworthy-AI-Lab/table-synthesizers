import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer


def create_test_dataframe():
    """Create a simple test DataFrame for testing CART"""
    np.random.seed(42)  # For reproducible tests
    
    df = pd.DataFrame({
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'binary_col': np.random.choice([0, 1], 100),
        'float_col': np.random.uniform(0, 1, 100)
    })
    
    return df


def test_CART_initialization():
    """Test CART model initialization"""
    config = {"max_depth": 5, "random_state": 42}
    synthesizer = TableSynthesizer('CART', config)
    assert synthesizer.model is not None
    print("CART model initialized successfully")


def test_CART_dataframe_support():
    """Test CART with DataFrame input"""
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Configuration for CART
    config = {"max_depth": 5, "random_state": 42}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer('CART', config)
    
    # Test fitting with DataFrame (CART only supports DataFrame)
    synthesizer.fit(df)
    
    # Test tensor output
    n_samples = 10
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
    
    print("CART DataFrame test passed successfully!")


def test_CART_different_data_types():
    """Test CART with different data types"""
    # Create DataFrame with various data types
    np.random.seed(42)
    df = pd.DataFrame({
        'int_col': np.random.randint(1, 10, 50),
        'float_col': np.random.uniform(0, 1, 50),
        'categorical_col': np.random.choice(['X', 'Y', 'Z'], 50),
        'binary_col': np.random.choice([True, False], 50),
    })
    
    config = {"max_depth": 3, "random_state": 42}
    synthesizer = TableSynthesizer('CART', config)
    
    # Fit and generate
    synthesizer.fit(df)
    synthetic_df = synthesizer.sample(n=20, return_dataframe=True)
    
    assert synthetic_df.shape[0] == 20
    assert synthetic_df.shape[1] == df.shape[1]
    
    # Check that columns exist
    for col in df.columns:
        assert col in synthetic_df.columns
    
    print("CART different data types test passed!")


if __name__ == "__main__":
    test_CART_initialization()
    test_CART_dataframe_support()
    test_CART_different_data_types()