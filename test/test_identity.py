import pytest
import sys
import os

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from stg.tableSynthesizer import TableSynthesizer
from test_data.data_info import load_and_process_data
from utils import run_sandbox_dataset_test

@pytest.fixture
def data():
    return load_and_process_data()

def test_identity_return(data):
    model = 'Identity'
    config = {"bootstrap": False}
    
    dataloaders, data_infos = data
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        synthesizer = TableSynthesizer(model, config, data_info)
        synthesizer.fit(dataloader)
        sampled_data = synthesizer.sample(n=len(dataloader))
        print("*"*40)
        print(sampled_data.shape, type(sampled_data), len(dataloader))
        print("*"*40)
        
        assert sampled_data.shape[0] == data_info['original_size'], "Sampled data length mismatch"
        #assert isinstance(sampled_data, type(dataloader.dataset)), "Sampled data type mismatch"

def test_identity_dataframe_support(data):
    """Test Identity with DataFrame input using shared utility"""
    from utils import test_dataframe_support
    
    config = {"bootstrap": False}
    test_dataframe_support('Identity', config, n_samples=10)


def test_identity_sandbox_insurance():
    """Test Identity on insurance dataset"""
    run_sandbox_dataset_test('Identity', 'insurance', n_samples=50, sample_ratio=0.2)


def test_identity_sandbox_adult():
    """Test Identity on adult dataset"""
    run_sandbox_dataset_test('Identity', 'adult', n_samples=50, sample_ratio=0.1)


def test_identity_sandbox_covtype():
    """Test Identity on covtype dataset"""
    run_sandbox_dataset_test('Identity', 'covtype', n_samples=50, sample_ratio=0.01)


