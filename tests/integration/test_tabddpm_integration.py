import pytest
import sys
import os
import torch

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer
from test_data.data_info import load_and_process_data
from utils import run_sandbox_dataset_test, load_sandbox_datasets

pytestmark = pytest.mark.gpu

@pytest.fixture
def data():
    return load_and_process_data()

def test_TabDDPM(data):
    model = 'TabDDPM'
    config = {"steps": 100, "num_timesteps": 10, "batch_size": 64}  # Reduced steps for faster testing
    
    dataloaders, data_infos = data
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        synthesizer = TableSynthesizer(model, config, data_info)
        #print("data_info:")
        #print(data_info)
        synthesizer.fit(dataloader)
        sampled_data = synthesizer.sample(n=data_info['original_size'])
        print("*"*40)
        print(sampled_data.shape, type(sampled_data), len(dataloader))
        print("*"*40)
        
        assert sampled_data.shape[0] == data_info['original_size'], "Sampled data length mismatch"
        assert isinstance(sampled_data, torch.Tensor), "Sampled data must be tensor!"
        #assert isinstance(sampled_data, type(dataloader.dataset)), "Sampled data type mismatch"


def test_TabDDPM_sandbox_insurance():
    """Test TabDDPM on insurance dataset with mixed categorical/numerical data"""
    run_sandbox_dataset_test('TabDDPM', 'insurance', n_samples=50, sample_ratio=0.1)


def test_TabDDPM_sandbox_titanic():
    """Test TabDDPM on Titanic dataset with mixed categorical/numerical data"""
    run_sandbox_dataset_test('TabDDPM', 'Titanic', n_samples=50, sample_ratio=0.2)


def test_TabDDPM_sandbox_bean():
    """Test TabDDPM on Bean dataset for classification"""
    run_sandbox_dataset_test('TabDDPM', 'Bean', n_samples=50, sample_ratio=0.05)

if __name__ == "__main__":
    test_TabDDPM(load_and_process_data())

    # Test sandbox datasets
    test_TabDDPM_sandbox_insurance()
    test_TabDDPM_sandbox_titanic()
    test_TabDDPM_sandbox_bean()
