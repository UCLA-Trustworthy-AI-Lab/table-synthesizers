import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer
from utils import run_sandbox_dataset_test, load_sandbox_datasets

# Check if SMOTE is available
try:
    from stg.tableSynthesizer import DEFAULT_MODELS
    SMOTE_AVAILABLE = 'SMOTE' in DEFAULT_MODELS
except ImportError:
    SMOTE_AVAILABLE = False


def create_test_dataframe_classification():
    """Create a test DataFrame for classification tasks"""
    np.random.seed(42)  # For reproducible tests
    
    # Create imbalanced dataset for SMOTE
    df = pd.DataFrame({
        'numeric_col1': np.random.randn(100),
        'numeric_col2': np.random.uniform(0, 1, 100),
        'categorical_col': np.random.choice(['X', 'Y'], 100),
        'target': np.concatenate([
            np.repeat('Class_A', 70),  # Majority class
            np.repeat('Class_B', 20),  # Minority class 1
            np.repeat('Class_C', 10)   # Minority class 2
        ])
    })
    
    # Shuffle the dataframe
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.mark.skipif(not SMOTE_AVAILABLE, reason="SMOTE not available due to missing dependencies")
def test_SMOTE_initialization():
    """Test SMOTE model initialization"""
    config = {"k_neighbors": 3, "random_state": 42}
    synthesizer = TableSynthesizer('SMOTE', config)
    assert synthesizer.model is not None
    print("SMOTE model initialized successfully")


@pytest.mark.skipif(not SMOTE_AVAILABLE, reason="SMOTE not available due to missing dependencies")
def test_SMOTE_dataframe_support():
    """Test SMOTE with DataFrame input"""
    # Create test DataFrame
    df = create_test_dataframe_classification()
    
    # Configuration for SMOTE
    config = {"k_neighbors": 3, "target_column": "target", "random_state": 42}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer('SMOTE', config)
    
    # Test fitting with DataFrame (SMOTE only supports DataFrame)
    synthesizer.fit(df)
    
    # Test tensor output
    n_samples = 20
    sampled_tensor = synthesizer.sample(n=n_samples)
    # SMOTE may generate slightly different count due to proportional class distribution
    assert abs(sampled_tensor.shape[0] - n_samples) <= 2, f"Expected ~{n_samples} samples, got {sampled_tensor.shape[0]}"
    assert sampled_tensor.shape[1] == df.shape[1], f"Expected {df.shape[1]} features, got {sampled_tensor.shape[1]}"
    
    # Test DataFrame output
    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    assert abs(sampled_df.shape[0] - n_samples) <= 2, f"Expected ~{n_samples} samples, got {sampled_df.shape[0]}"
    assert isinstance(sampled_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(sampled_df)}"
    
    # Check column names match
    expected_cols = set(df.columns)
    actual_cols = set(sampled_df.columns)
    assert expected_cols == actual_cols, f"Column names mismatch. Expected: {expected_cols}, Got: {actual_cols}"
    
    print("SMOTE DataFrame test passed successfully!")


@pytest.mark.skipif(not SMOTE_AVAILABLE, reason="SMOTE not available due to missing dependencies")
def test_SMOTE_auto_target_detection():
    """Test SMOTE with automatic target column detection"""
    df = create_test_dataframe_classification()
    
    # Don't specify target column - should auto-detect last column
    config = {"k_neighbors": 3, "random_state": 42}
    synthesizer = TableSynthesizer('SMOTE', config)
    
    # Fit and generate
    synthesizer.fit(df)
    synthetic_df = synthesizer.sample(n=15, return_dataframe=True)
    
    assert abs(synthetic_df.shape[0] - 15) <= 2  # Allow slight variation due to class proportions
    assert synthetic_df.shape[1] == df.shape[1]
    
    # Should have same columns
    assert set(df.columns) == set(synthetic_df.columns)
    
    print("SMOTE auto target detection test passed!")


@pytest.mark.skipif(not SMOTE_AVAILABLE, reason="SMOTE not available due to missing dependencies") 
def test_SMOTE_with_mixed_features():
    """Test SMOTE with mixed feature types"""
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric1': np.random.randn(80),
        'numeric2': np.random.uniform(0, 10, 80),
        'categorical1': np.random.choice(['A', 'B', 'C'], 80),
        'categorical2': np.random.choice(['Red', 'Blue'], 80),
        'target': np.concatenate([
            np.repeat('Positive', 50),
            np.repeat('Negative', 30)
        ])
    })
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    config = {
        "target_column": "target",
        "categorical_features": ["categorical1", "categorical2"], 
        "k_neighbors": 3,
        "random_state": 42
    }
    
    synthesizer = TableSynthesizer('SMOTE', config)
    synthesizer.fit(df)
    
    synthetic_df = synthesizer.sample(n=25, return_dataframe=True)
    
    # Verify output
    assert abs(synthetic_df.shape[0] - 25) <= 2  # Allow slight variation due to class proportions
    assert synthetic_df.shape[1] == 5  # 4 features + 1 target
    assert all(col in synthetic_df.columns for col in df.columns)
    
    print("SMOTE mixed features test passed!")


@pytest.mark.skipif(not SMOTE_AVAILABLE, reason="SMOTE not available due to missing dependencies")
def test_SMOTE_sandbox_adult():
    """Test SMOTE on adult dataset (classification task)"""
    run_sandbox_dataset_test('SMOTE', 'adult', n_samples=100, sample_ratio=0.05)


@pytest.mark.skipif(not SMOTE_AVAILABLE, reason="SMOTE not available due to missing dependencies") 
def test_SMOTE_sandbox_covtype():
    """Test SMOTE on covtype dataset (multi-class classification)"""
    run_sandbox_dataset_test('SMOTE', 'covtype', n_samples=100, sample_ratio=0.01)


def test_SMOTE_availability():
    """Test if SMOTE is properly detected as available or not"""
    if SMOTE_AVAILABLE:
        print("SMOTE is available and registered in DEFAULT_MODELS")
    else:
        print("SMOTE is not available due to missing dependencies (expected)")
        pytest.skip("SMOTE not available due to missing dependencies")


if __name__ == "__main__":
    test_SMOTE_availability()
    if SMOTE_AVAILABLE:
        test_SMOTE_initialization()
        test_SMOTE_dataframe_support() 
        test_SMOTE_auto_target_detection()
        test_SMOTE_with_mixed_features()
        
        # Test sandbox datasets
        test_SMOTE_sandbox_adult()
        test_SMOTE_sandbox_covtype()