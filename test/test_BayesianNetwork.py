import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from stg.tableSynthesizer import TableSynthesizer

# Check if BayesianNetwork is available
try:
    from stg.tableSynthesizer import DEFAULT_MODELS
    import importlib.util
    BAYESIANNETWORK_AVAILABLE = 'BayesianNetwork' in DEFAULT_MODELS and importlib.util.find_spec('synthcity') is not None
except ImportError:
    BAYESIANNETWORK_AVAILABLE = False


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


@pytest.mark.skipif(not BAYESIANNETWORK_AVAILABLE, reason="BayesianNetwork not available due to missing dependencies")
def test_BayesianNetwork_initialization():
    """Test BayesianNetwork model initialization"""
    config = {}
    synthesizer = TableSynthesizer('BayesianNetwork', config)
    assert synthesizer.model is not None
    print("BayesianNetwork model initialized successfully")


@pytest.mark.skipif(not BAYESIANNETWORK_AVAILABLE, reason="BayesianNetwork not available due to missing dependencies")
def test_BayesianNetwork_dataframe_support():
    """Test BayesianNetwork with DataFrame input"""
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Configuration for BayesianNetwork
    config = {}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer('BayesianNetwork', config)
    
    # Test fitting with DataFrame (BayesianNetwork only supports DataFrame)
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
    
    print("BayesianNetwork DataFrame test passed successfully!")


@pytest.mark.skipif(not BAYESIANNETWORK_AVAILABLE, reason="BayesianNetwork not available due to missing dependencies")
def test_BayesianNetwork_different_data_types():
    """Test BayesianNetwork with different data types"""
    np.random.seed(42)
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 40),
        'float_col': np.random.randn(40),
        'category_col': np.random.choice(['X', 'Y', 'Z'], 40),
        'binary_col': np.random.choice(['Yes', 'No'], 40)
    })
    
    config = {}
    synthesizer = TableSynthesizer('BayesianNetwork', config)
    synthesizer.fit(df)
    
    synthetic_df = synthesizer.sample(n=15, return_dataframe=True)
    
    # Verify output
    assert synthetic_df.shape[0] == 15
    assert synthetic_df.shape[1] == 4
    assert all(col in synthetic_df.columns for col in df.columns)
    
    print("BayesianNetwork different data types test passed!")


def test_BayesianNetwork_availability():
    """Test if BayesianNetwork is properly detected as available or not"""
    if BAYESIANNETWORK_AVAILABLE:
        print("BayesianNetwork is available and registered in DEFAULT_MODELS")
    else:
        print("BayesianNetwork is not available due to missing synthcity dependencies (expected)")
        pytest.skip("BayesianNetwork not available due to missing dependencies")


if __name__ == "__main__":
    if not BAYESIANNETWORK_AVAILABLE:
        print("BayesianNetwork not available due to missing synthcity dependencies (expected). Skipping tests.")
        sys.exit(0)
        
    test_BayesianNetwork_availability()
    test_BayesianNetwork_initialization()
    test_BayesianNetwork_dataframe_support() 
    test_BayesianNetwork_different_data_types()