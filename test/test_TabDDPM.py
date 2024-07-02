import pytest
import sys
import os

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from stg.tableSynthesizer import TableSynthesizer
from test_data.data_info import load_and_process_data

@pytest.fixture
def data():
    return load_and_process_data()

def test_TabDDPM(data):
    model = 'TabDDPM'
    config = {}
    
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
        #assert isinstance(sampled_data, type(dataloader.dataset)), "Sampled data type mismatch"

if __name__ == "__main__":
    test_TabDDPM(load_and_process_data())
