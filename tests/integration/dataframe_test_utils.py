"""
Shared utilities for testing DataFrame input functionality across all synthesizers
"""
import pandas as pd
import numpy as np
import torch
import sys
import os

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


def create_test_dataframe():
    """Create a simple test DataFrame for testing DataFrame input functionality"""
    np.random.seed(42)  # For reproducible tests

    df = pd.DataFrame({
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'binary_col': np.random.choice([0, 1], 100),
        'float_col': np.random.uniform(0, 1, 100)
    })

    return df


def test_dataframe_support(model_name, config=None, n_samples=10):
    """
    Generic test function for DataFrame support that can be used by any synthesizer

    Args:
        model_name (str): Name of the synthesizer model
        config (dict): Configuration parameters for the model
        n_samples (int): Number of samples to generate

    Returns:
        bool: True if test passes
    """
    from stg.tableSynthesizer import TableSynthesizer

    # Create test DataFrame
    df = create_test_dataframe()

    # Use default config if none provided
    if config is None:
        config = {"epochs": 1, "batch_size": 32}

    # Initialize synthesizer
    synthesizer = TableSynthesizer(model_name, config)

    # Test fitting with DataFrame
    batch_size = config.get('batch_size', 32)
    synthesizer.fit(df, batch_size=batch_size)

    # Test tensor output
    sampled_tensor = synthesizer.sample(n=n_samples)
    assert sampled_tensor.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_tensor.shape[0]}"
    assert isinstance(sampled_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(sampled_tensor)}"

    # Test DataFrame output
    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    assert sampled_df.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_df.shape[0]}"
    assert isinstance(sampled_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(sampled_df)}"

    # Check column names (more flexible since encoding might change order)
    expected_cols = set(['numeric_col', 'categorical_col', 'binary_col', 'float_col'])
    actual_cols = set(sampled_df.columns)
    print(f"Expected columns: {expected_cols}")
    print(f"Actual columns: {actual_cols}")
    assert expected_cols == actual_cols, f"Column names mismatch. Expected: {expected_cols}, Got: {actual_cols}"

    print(f"{model_name} DataFrame test passed successfully!")
    return True