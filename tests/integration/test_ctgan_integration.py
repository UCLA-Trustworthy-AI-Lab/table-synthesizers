import pytest
import sys
import os
import torch

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer
from test_data.data_info import load_and_process_data
from utils import run_sandbox_dataset_test

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Skip sandbox tests if no external dataset directory is available.
# Set DATASET_PATH env var or place CSVs in tests/integration/test_data/sandbox_datasets/
# ---------------------------------------------------------------------------
_SANDBOX_DIR = os.environ.get(
    'DATASET_PATH',
    os.path.join(os.path.dirname(__file__), 'test_data', 'sandbox_datasets')
)
_HAS_SANDBOX = os.path.isdir(_SANDBOX_DIR) and bool(
    [f for f in os.listdir(_SANDBOX_DIR) if f.endswith('.csv')] if os.path.isdir(_SANDBOX_DIR) else []
)

skip_if_no_sandbox = pytest.mark.skipif(
    not _HAS_SANDBOX,
    reason=(
        f"Sandbox datasets not found at '{_SANDBOX_DIR}'. "
        "Set DATASET_PATH env var or add CSVs to tests/integration/test_data/sandbox_datasets/"
    ),
)

@pytest.fixture
def data():
    return load_and_process_data()

def test_CTGAN(data):
    model = 'CTGAN'
    config = {"epochs": 1, "batch_size": 32, "pac": 1, "generator_lr": 0.0002, "discriminator_lr": 0.0002}  # Minimal config for fast testing
    
    dataloaders, data_infos = data
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        synthesizer = TableSynthesizer(model, config, data_info)
        print("data_info:")
        print(data_info)
        synthesizer.fit(dataloader)
        sampled_data = synthesizer.sample(n=data_info['original_size'])
        print("*"*40)
        print(sampled_data.shape, type(sampled_data), len(dataloader))
        print("*"*40)
        
        assert sampled_data.shape[0] == data_info['original_size'], "Sampled data length mismatch"
        assert isinstance(sampled_data, torch.Tensor), "Sampled data must be tensor!"
        #assert isinstance(sampled_data, type(dataloader.dataset)), "Sampled data type mismatch"

def test_CTGAN_dataframe_support(data):
    """Test CTGAN with DataFrame input using shared utility"""
    from utils import test_dataframe_support
    
    config = {"epochs": 1, "batch_size": 32, "embedding_dim": 64, "pac": 1}  # Reduced epochs for testing
    test_dataframe_support('CTGAN', config, n_samples=10)


@skip_if_no_sandbox
def test_PATECTGAN_sandbox_insurance():
    """Test PATECTGAN on insurance dataset (skipped if sandbox data unavailable)"""
    run_sandbox_dataset_test('PATECTGAN', 'insurance', n_samples=50, sample_ratio=0.1)


@skip_if_no_sandbox
def test_PATECTGAN_sandbox_titanic():
    """Test PATECTGAN on Titanic dataset (skipped if sandbox data unavailable)"""
    run_sandbox_dataset_test('PATECTGAN', 'Titanic', n_samples=50, sample_ratio=0.2)


@skip_if_no_sandbox
def test_PATECTGAN_sandbox_bean():
    """Test PATECTGAN on Bean dataset (skipped if sandbox data unavailable)"""
    run_sandbox_dataset_test('PATECTGAN', 'Bean', n_samples=50, sample_ratio=0.05)

if __name__ == "__main__":
    test_CTGAN(load_and_process_data())

    # Test sandbox datasets with PATECTGAN (CTGAN replacement)
    test_PATECTGAN_sandbox_insurance()
    test_PATECTGAN_sandbox_titanic()
    test_PATECTGAN_sandbox_bean()
