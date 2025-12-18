import pytest
import pandas as pd
import numpy as np
import torch
from stg.DPCART.dpcart_synthesizer import DPCARTSynthesizer

def test_dpcart_initialization():
    """Test that DPCART can be initialized."""
    model = DPCARTSynthesizer()
    assert model.trained_data is None

def test_dpcart_fit_and_sample(sample_data):
    """Test DPCART training and sampling."""
    model = DPCARTSynthesizer(max_depth=5, epsilon_per_tree=1.0)
    
    model.fit(sample_data)
    
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]

def test_dpcart_save_load(sample_data, tmp_path):
    """Test saving and loading."""
    model = DPCARTSynthesizer()
    model.fit(sample_data)
    
    import pickle
    save_path = tmp_path / "dpcart.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(save_path, 'rb') as f:
        loaded_model = pickle.load(f)
        
    assert loaded_model.trained_data is not None
