import pytest
import pandas as pd
import numpy as np
import torch
import shutil
import os
from stg.AutoDiff.autodiff_synthesizer import AutoDiffSynthesizer, AUTODIFF_AVAILABLE

@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff dependencies not installed")
def test_autodiff_initialization():
    """Test that AutoDiff can be initialized."""
    model = AutoDiffSynthesizer(n_epochs=1, diff_n_epochs=1)
    assert model.n_epochs == 1

@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff dependencies not installed")
def test_autodiff_fit_and_sample(sample_data):
    """Test AutoDiff training and sampling."""
    model = AutoDiffSynthesizer(n_epochs=1, diff_n_epochs=1)
    
    # Fit model matches base class signature but overrides behavior
    model.fit(sample_data)
    
    # Sample
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]

@pytest.mark.skipif(not AUTODIFF_AVAILABLE, reason="AutoDiff dependencies not installed")
def test_autodiff_save_load(sample_data, tmp_path):
    """Test saving and loading."""
    model = AutoDiffSynthesizer(n_epochs=1, diff_n_epochs=1)
    model.fit(sample_data)
    
    # AutoDiff has complex state (ds, score). 
    # Check if we can pickle the object for now as a basic persistence test.
    import pickle
    save_path = tmp_path / "autodiff.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(save_path, 'rb') as f:
        loaded_model = pickle.load(f)
        
    assert loaded_model.ds is not None
