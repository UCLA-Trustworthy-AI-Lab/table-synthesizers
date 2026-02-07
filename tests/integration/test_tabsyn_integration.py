import pytest
import sys
import os
import pandas as pd
import numpy as np
import time

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer
from utils import run_sandbox_dataset_test

# Check if TabSyn is actually available (dependencies installed)
try:
    from stg.TabSyn.tabsyn_synthesizer import TABSYN_AVAILABLE
except ImportError:
    TABSYN_AVAILABLE = False


def create_test_dataframe():
    """Create a test DataFrame"""
    np.random.seed(42)  # For reproducible tests
    
    df = pd.DataFrame({
        'numeric_col1': np.random.randn(50),
        'numeric_col2': np.random.uniform(0, 1, 50),
        # Encode categorical as small integers to avoid high-cardinality object dtypes
        'categorical_col': np.random.choice([0, 1, 2], 50),
        'binary_col': np.random.choice([0, 1], 50)
    })
    
    return df


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn not available due to missing dependencies")
def test_TabSyn_initialization():
    """Test TabSyn model initialization"""
    config = {"dataset_name": "test_dataset"}
    t0 = time.time(); print("[TEST][TabSyn] init start", flush=True)
    synthesizer = TableSynthesizer('TabSyn', config)
    assert synthesizer.model is not None
    print(f"[TEST][TabSyn] init done in {time.time()-t0:.2f}s", flush=True)


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn not available due to missing dependencies")
def test_TabSyn_dataframe_support():
    """Test TabSyn with DataFrame input"""
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Configuration for TabSyn
    config = {"dataset_name": "test_tabsyn_df"}
    
    # Initialize synthesizer
    init_t = time.time(); print("[TEST][TabSyn] creating synthesizer...", flush=True)
    synthesizer = TableSynthesizer('TabSyn', config)
    print(f"[TEST][TabSyn] synthesizer created in {time.time()-init_t:.2f}s", flush=True)
    
    # Test fitting with DataFrame (TabSyn only supports DataFrame)
    fit_t = time.time(); print("[TEST][TabSyn] calling fit(...)", flush=True)
    synthesizer.fit(df)
    print(f"[TEST][TabSyn] fit done in {time.time()-fit_t:.2f}s", flush=True)
    
    # Test tensor output
    n_samples = 20
    samp_t = time.time(); print("[TEST][TabSyn] sampling tensor...", flush=True)
    sampled_tensor = synthesizer.sample(n=n_samples)
    print(f"[TEST][TabSyn] tensor sample done in {time.time()-samp_t:.2f}s", flush=True)
    assert sampled_tensor.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_tensor.shape[0]}"
    assert sampled_tensor.shape[1] == df.shape[1], f"Expected {df.shape[1]} features, got {sampled_tensor.shape[1]}"
    
    # Test DataFrame output
    samp_df_t = time.time(); print("[TEST][TabSyn] sampling dataframe...", flush=True)
    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    print(f"[TEST][TabSyn] dataframe sample done in {time.time()-samp_df_t:.2f}s", flush=True)
    assert sampled_df.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_df.shape[0]}"
    assert isinstance(sampled_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(sampled_df)}"
    
    # Check column names match
    expected_cols = set(df.columns)
    actual_cols = set(sampled_df.columns)
    assert expected_cols == actual_cols, f"Column names mismatch. Expected: {expected_cols}, Got: {actual_cols}"
    
    print("[TEST][TabSyn] DataFrame test passed successfully!", flush=True)


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn not available due to missing dependencies")
def test_TabSyn_different_data_types():
    """Test TabSyn with different data types"""
    np.random.seed(42)
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 40),
        'float_col': np.random.randn(40),
        # Encode categories to small integers to guarantee feasible train/val split
        'category_col': np.random.choice([0, 1, 2], 40),
        'binary_col': np.random.choice([0, 1], 40)
    })
    
    config = {"dataset_name": "test_tabsyn_mixed"}
    init_t = time.time(); print("[TEST][TabSyn] creating synthesizer (mixed types)...", flush=True)
    synthesizer = TableSynthesizer('TabSyn', config)
    print(f"[TEST][TabSyn] synthesizer created in {time.time()-init_t:.2f}s", flush=True)
    fit_t = time.time(); print("[TEST][TabSyn] calling fit(...) (mixed types)", flush=True)
    synthesizer.fit(df)
    print(f"[TEST][TabSyn] fit done in {time.time()-fit_t:.2f}s", flush=True)
    samp_df_t = time.time(); print("[TEST][TabSyn] sampling dataframe (mixed types)...", flush=True)
    synthetic_df = synthesizer.sample(n=15, return_dataframe=True)
    print(f"[TEST][TabSyn] dataframe sample done in {time.time()-samp_df_t:.2f}s", flush=True)
    
    # Verify output
    assert synthetic_df.shape[0] == 15
    assert synthetic_df.shape[1] == 4
    assert all(col in synthetic_df.columns for col in df.columns)
    
    print("[TEST][TabSyn] different data types test passed!", flush=True)


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn not available due to missing dependencies")
def test_TabSyn_sandbox_insurance():
    """Test TabSyn on insurance dataset"""
    run_sandbox_dataset_test('TabSyn', 'insurance', n_samples=50, sample_ratio=0.1)


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn not available due to missing dependencies")
def test_TabSyn_sandbox_titanic():
    """Test TabSyn on Titanic dataset"""
    run_sandbox_dataset_test('TabSyn', 'Titanic', n_samples=50, sample_ratio=0.2)


def test_TabSyn_availability():
    """Test if TabSyn is properly detected as available or not"""
    if TABSYN_AVAILABLE:
        print("TabSyn is available and registered in DEFAULT_MODELS")
    else:
        print("TabSyn is not available due to missing dependencies (expected)")
        pytest.skip("TabSyn not available due to missing dependencies")


if __name__ == "__main__":
    test_TabSyn_availability()
    if TABSYN_AVAILABLE:
        test_TabSyn_initialization()
        test_TabSyn_dataframe_support() 
        test_TabSyn_different_data_types()