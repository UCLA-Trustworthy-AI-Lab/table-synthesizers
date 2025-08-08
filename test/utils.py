from syntest import DataHost
import torch
import itertools
import pandas as pd
import numpy as np

import dask.dataframe as dd
from lddp_optimus import TableTransformer

import sdmetrics
from sdmetrics.reports.single_table import QualityReport


# Generate all combinations of the parameter choices
n_num_choices = [5, 0] #0, 5
n_cat_choices = [5, 0] #0, 5
n_rows_choices = [1000]
missing_percent_choices = [0, 0.1] #0, 0.1, 0.5, 0.9

parameter_combinations = list(itertools.product(
    n_num_choices,
    n_cat_choices,
    n_rows_choices,
    missing_percent_choices
))

# Format the combinations into the desired structure
test_parameters = [
    (n_num, n_cat, n_rows, missing_percent)
    for n_num, n_cat, n_rows, missing_percent in parameter_combinations if (n_cat > 0 or n_num > 0)
]


def get_data(data_source):
    """
        Get data and config from syntest
    """
    datasets = {}
    host = DataHost()
    if data_source in ['real', 'both']:
        datasets.update(host.get_all_datasets())
    if data_source in ['fake', 'both']:
        for n_num, n_cat, n_rows, missing_percent in test_parameters:
            datasets[f'fake_{n_num}_{n_cat}_{n_rows}_{missing_percent}'] = host.get_fake_dataset(n_num, n_cat, n_rows, missing_percent)
    if data_source not in ['real', 'fake','both']:
        real, fake, config = host.get_dataset(data_source)
        datasets[data_source] = {'real':real, 'fake':fake, 'config':config}

    return datasets

def get_model_config(model_name):
    """
        Return model specific mapping + training parameters
    """
    if model_name == 'Identity':
        model_config = {'mapping':{
            "continuous": "mm",
            "categorical": "ohe",
            "pii": "pii",
            "datetime": "dt"
        },"bootstrap": False}
    if model_name == 'CTGAN':
        model_config = {'mapping':{
            "continuous": "mm",
            "categorical": "ohe",
            "pii": "pii",
            "datetime": "dt"
        },"epochs": 300, "batch_size":500}
    if model_name == "TabDDPM":
        model_config = {'mapping':{
            "continuous": "gqt",
            "categorical": "le",
            "pii": "pii",
            "datetime": "dt"
        },"steps": 10000,
        "batch_size":1024}

    return model_config

def transform_data(real, config, mapping, batch_size = 300, **kwargs):
    """
        Transform the data above to tensor loader using lddp
    """
    ddf = dd.from_pandas(real)

    meta = config['meta']
    for c in meta:
        if meta[c] == "numerical":
            meta[c] = 'continuous'

    tableTransformer = TableTransformer(meta, mapping)
    #tableTransformer.set_local_cluaster()
    transformed_ddf = tableTransformer.fit_transform(ddf)
    print("Head of transformed data:")
    print(transformed_ddf.head())

    tensor_loader = tableTransformer.ddf_to_tensor_loader_naive(transformed_ddf, batch_size)


    for batch in tensor_loader:
        print(batch.shape, type(batch))
        break 

    #tableTransformer.close_local_clusters()

    return tensor_loader, tableTransformer.get_transform_info(), tableTransformer


def validate_output(real_df, sampled_data, data_info, config, tableTransformer):
    """
        Confirm the size/type of synthetic data is good. TODO: add automatic quality check.
    """
    assert sampled_data.shape[0] == data_info['original_size'], "Sampled data length mismatch"
    assert isinstance(sampled_data, torch.Tensor), "Sampled data must be tensor!"

    print(config['meta'])

    sampled_df = tableTransformer.tensor_to_ddf(sampled_data)
    print("sampled_df:")
    print(sampled_df.compute())
    sampled_df.compute().to_csv("sampled_df.csv",index=False)
    synthetic_df = tableTransformer.inverse_transform(sampled_df).compute()
    metadata = {'columns':{}}
    for c in config['meta']:
        metadata['columns'][c] = {}
        metadata['columns'][c]['sdtype'] = 'categorical' if config['meta'][c] == 'categorical' else 'numerical'

    quality_report = QualityReport()
    quality_report.generate(real_df, synthetic_df, metadata)
    print(quality_report.get_details('Column Shapes'))


    print(real_df.head())
    print(synthetic_df.head())


def create_test_dataframe():
    """Create a simple test DataFrame for testing DataFrame input functionality"""
    np.random.seed(42)  # For reproducible tests
    
    df = pd.DataFrame({
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'binary_col': np.random.choice([0, 1], 100),
        'float_col': np.random.uniform(0, 1, 100)
    })
    
    return df


def test_dataframe_support(model_name, config=None, n_samples=10):
    """
    Generic test function for DataFrame support that can be used by any synthesizer
    
    Args:
        model_name (str): Name of the synthesizer model
        config (dict): Configuration parameters for the model
        n_samples (int): Number of samples to generate
    
    Returns:
        bool: True if test passes
    """
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from stg.tableSynthesizer import TableSynthesizer
    
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Use default config if none provided
    if config is None:
        config = {"epochs": 1, "batch_size": 32}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer(model_name, config)
    
    # Test fitting with DataFrame
    synthesizer.fit(df, batch_size=32)
    
    # Test tensor output
    sampled_tensor = synthesizer.sample(n=n_samples)
    assert sampled_tensor.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_tensor.shape[0]}"
    assert isinstance(sampled_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(sampled_tensor)}"
    
    # Test DataFrame output
    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    assert sampled_df.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_df.shape[0]}"
    assert isinstance(sampled_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(sampled_df)}"
    assert list(sampled_df.columns) == ['numeric_col', 'categorical_col', 'binary_col', 'float_col'], "Column names mismatch"
    
    print(f"{model_name} DataFrame test passed successfully!")
    return True

