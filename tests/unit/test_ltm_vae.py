import pytest
import pandas as pd
import numpy as np
import torch

pytestmark = pytest.mark.gpu

# Try to import LTM_VAE, handle case where it doesn't exist
try:
    from stg.LTM_VAE import LTMVAESynthesizer, LTM_VAE_AVAILABLE
except ImportError:
    LTMVAESynthesizer = None
    LTM_VAE_AVAILABLE = False


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'binary_col': np.random.choice([0, 1], 100),
    })


@pytest.mark.skipif(not LTM_VAE_AVAILABLE, reason="LTM-VAE not installed")
def test_ltm_vae_initialization():
    model = LTMVAESynthesizer(num_epochs=1)
    assert model.num_epochs == 1


@pytest.mark.skipif(not LTM_VAE_AVAILABLE, reason="LTM-VAE not installed")
def test_ltm_vae_fit_and_sample(sample_data):
    model = LTMVAESynthesizer(num_epochs=1, config_task='quick_test')

    model.fit(sample_data)

    samples = model.sample(10, return_dataframe=True)

    assert len(samples) == 10
    # LTM typically returns same columns
    assert samples.shape[1] == sample_data.shape[1]


def test_ltm_vae_availability():
    """Test if LTM-VAE is properly detected as available or not."""
    if LTM_VAE_AVAILABLE:
        print("LTM-VAE is available and can be imported")
    else:
        print("LTM-VAE is not available (expected - module not installed)")
        pytest.skip("LTM-VAE not available")
