import pytest
import pandas as pd
import numpy as np
import torch
from stg.CART.cart_synthesizer import CARTSynthesizer

def test_cart_initialization():
    """Test that CART can be initialized."""
    model = CARTSynthesizer()
    assert model.trained_data is None

def test_cart_fit_and_sample(sample_data):
    """Test CART training and sampling."""
    model = CARTSynthesizer(max_depth=5)
    
    model.fit(sample_data)
    
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]

def test_cart_save_load(sample_data, tmp_path):
    """Test saving and loading."""
    model = CARTSynthesizer()
    model.fit(sample_data)
    
    import pickle
    save_path = tmp_path / "cart.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(save_path, 'rb') as f:
        loaded_model = pickle.load(f)
        
    assert loaded_model.trained_data is not None
