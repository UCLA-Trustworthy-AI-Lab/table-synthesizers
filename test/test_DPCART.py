import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from stg.tableSynthesizer import TableSynthesizer


def create_test_dataframe():
    """Create a simple test DataFrame for testing DPCART"""
    np.random.seed(42)  # For reproducible tests
    
    df = pd.DataFrame({
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'binary_col': np.random.choice([0, 1], 100),
        'float_col': np.random.uniform(0, 1, 100)
    })
    
    return df


def test_DPCART_initialization():
    """Test DPCART model initialization"""
    config = {"max_depth": 5, "epsilon_per_tree": 1.0, "random_state": 42}
    synthesizer = TableSynthesizer('DPCART', config)
    assert synthesizer.model is not None
    print("DPCART model initialized successfully")


def test_DPCART_dataframe_support():
    """Test DPCART with DataFrame input"""
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Configuration for DPCART with privacy budget
    config = {"max_depth": 5, "epsilon_per_tree": 2.0, "random_state": 42}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer('DPCART', config)
    
    # Test fitting with DataFrame (DPCART only supports DataFrame)
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
    
    print("DPCART DataFrame test passed successfully!")


def test_DPCART_privacy_parameters():
    """Test DPCART with different privacy parameters"""
    df = create_test_dataframe()
    
    # Test with high privacy (low epsilon)
    config_high_privacy = {"epsilon_per_tree": 0.1, "random_state": 42}
    synthesizer_hp = TableSynthesizer('DPCART', config_high_privacy)
    synthesizer_hp.fit(df)
    synthetic_hp = synthesizer_hp.sample(n=10, return_dataframe=True)
    
    # Test with low privacy (high epsilon)
    config_low_privacy = {"epsilon_per_tree": 10.0, "random_state": 42}
    synthesizer_lp = TableSynthesizer('DPCART', config_low_privacy)
    synthesizer_lp.fit(df)
    synthetic_lp = synthesizer_lp.sample(n=10, return_dataframe=True)
    
    # Both should produce valid output
    assert synthetic_hp.shape == (10, 4)
    assert synthetic_lp.shape == (10, 4)
    
    print("DPCART privacy parameters test passed!")


if __name__ == "__main__":
    test_DPCART_initialization()
    test_DPCART_dataframe_support()
    test_DPCART_privacy_parameters()