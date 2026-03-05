import pytest
import sys
import os

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer
from test_data.data_info import load_and_process_data

pytestmark = pytest.mark.gpu

@pytest.fixture
def data():
    return load_and_process_data()

def test_TVAE_initialization(data):
    """Test TVAE model initialization"""
    model = 'TVAE'
    config = {"epochs": 1, "batch_size": 32}
    
    dataloaders, data_infos = data
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        synthesizer = TableSynthesizer(model, config, data_info)
        assert synthesizer.model is not None
        print(f"TVAE model initialized successfully for dataset {i}")

def test_TVAE_training(data):
    """Test TVAE model training"""
    model = 'TVAE'
    config = {"epochs": 2, "batch_size": 32, "embedding_dim": 64, "compress_dims": (64, 32), "decompress_dims": (32, 64)}
    
    dataloaders, data_infos = data
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        synthesizer = TableSynthesizer(model, config, data_info)
        synthesizer.fit(dataloader)
        print(f"TVAE model trained successfully for dataset {i}")

def test_TVAE_sampling(data):
    """Test TVAE model sampling"""
    model = 'TVAE'
    config = {"epochs": 2, "batch_size": 32, "embedding_dim": 64, "compress_dims": (64, 32), "decompress_dims": (32, 64)}
    
    dataloaders, data_infos = data
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        synthesizer = TableSynthesizer(model, config, data_info)
        synthesizer.fit(dataloader)
        sampled_data = synthesizer.sample(n=10)
        print(f"TVAE sampling successful for dataset {i}, shape: {sampled_data.shape}")
        assert sampled_data.shape[0] == 10, "Sampled data length mismatch"
        assert sampled_data.shape[1] == data_info['encoded_width'], "Sampled data width mismatch"

def test_TVAE_dataframe_support(data):
    """Test TVAE with DataFrame input using shared utility"""
    from utils import test_dataframe_support
    
    config = {"epochs": 1, "batch_size": 32, "embedding_dim": 64, "compress_dims": (64, 32), "decompress_dims": (32, 64)}
    test_dataframe_support('TVAE', config, n_samples=10)