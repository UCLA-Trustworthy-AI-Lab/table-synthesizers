import pytest

from utils import *

# Add src to the path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from stg.tableSynthesizer import TableSynthesizer


@pytest.mark.parametrize("data_source, model_name",[
    "real", "Identity"
])
def test_generation(data_source, model_name):
    """
        Data_source: real, fake, both
        model_name: used to pick model from tableSynthesizer and choose corresponding validation method. 
    """
    # 1, get_data
    datasets = get_data(data_source)

    # 2, add model-specific config 
    model_config = get_model_config(model_name)

    for dataset, elements in datasets.items():
        real,config = elements['real'], elements['config']
        config.update(model_config)

        # 3, transform_data
        dataloader, data_info, tableTransformer = transform_data(real, config, **config)
        data_info['original_size'] = len(real)

        # 4, run generation
        synthesizer = TableSynthesizer(model_name, config, data_info)
        print(f"Dataset: {dataset}")
        print("data_info:")
        print(data_info)
        synthesizer.fit(dataloader)
        sampled_data = synthesizer.sample(n=data_info['original_size'])
        print("*"*40)
        print(sampled_data.shape, type(sampled_data), len(dataloader))
        print("*"*40)

        # 5, validate_output
        validate_output(real, sampled_data, data_info, config, tableTransformer)


# TODO: add model-specici test functions below
def not_test_bootstrap(data):
    model = 'Identity'
    config = {"bootstrap": True}
    desired_length = 123
    
    dataloaders, data_infos = data
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        synthesizer = TableSynthesizer(model, config, data_info)
        synthesizer.fit(dataloader)
        sampled_data = synthesizer.sample(n=desired_length)
        print("*"*40)
        print(sampled_data.shape, type(sampled_data), len(dataloader))
        print("*"*40)
        
        assert sampled_data.shape[0] == desired_length, "Sampled data length mismatch"
        #assert isinstance(sampled_data, type(dataloader.dataset)), "Sampled data type mismatch"


if __name__ == "__main__":
    test_generation("insurance", 'CTGAN')