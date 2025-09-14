import os
import pytest

from utils import load_sandbox_datasets, run_sandbox_dataset_test


def available_models():
    # Discover which core models are registered
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from stg.tableSynthesizer import DEFAULT_MODELS
    return set(DEFAULT_MODELS.keys())


# Collect all sandbox dataset names dynamically
_SANDBOX = load_sandbox_datasets()
_DATASET_NAMES = sorted(list(_SANDBOX.keys()))


@pytest.mark.parametrize("dataset_name", _DATASET_NAMES)
def test_all_sandbox_identity(dataset_name):
    # Always run Identity as a fast end-to-end pipeline check
    run_sandbox_dataset_test('Identity', dataset_name, n_samples=30, sample_ratio=0.05)


@pytest.mark.parametrize("dataset_name", _DATASET_NAMES)
def test_all_sandbox_tvae(dataset_name):
    # Run TVAE across all datasets if available (small epochs for speed)
    if 'TVAE' not in available_models():
        pytest.skip("TVAE not registered")
    cfg = {"epochs": 1, "batch_size": 64}
    # Reduce sample ratio for large datasets
    ratio = 0.01 if dataset_name.lower().startswith('covtype') else 0.1
    run_sandbox_dataset_test('TVAE', dataset_name, config=cfg, n_samples=30, sample_ratio=ratio)

