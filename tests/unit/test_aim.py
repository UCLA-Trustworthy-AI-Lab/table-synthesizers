import pytest
import pandas as pd
import numpy as np
import torch
import shutil
import os
from stg.AIM.AIM import AIM

def test_aim_initialization():
    """Test that AIM can be initialized."""
    model = AIM(mach,epsilon=0.5)
    assert model.epsilon == 0.5

def test_aim_fit_and_sample(sample_categorical_data):
    """Test AIM training and sampling."""
    model = AIM(epsilon=1.0, epochs=1)
    
    # Fit model
    model.fit(sample_categorical_data)
    
    # Sample
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_categorical_data.shape[1]
    
    # Check if samples are valid (roughly)
    # Since AIM is for discrete data, verify output is somewhat discrete or reasonable
    # For now, just checking shape and type is good enough for a basic test.

def test_aim_save_load(sample_categorical_data, tmp_path):
    """Test saving and loading the AIM model."""
    model = AIM(epsilon=1.0, epochs=1)
    model.fit(sample_categorical_data)
    
    save_path = tmp_path / "aim_checkpoint.pt"
    
    # Mocking save which usually happens via DataManager or manually
    # AIM's save/load might leverage torch.save/load on get_state directly or via BaseSynthesizer
    # Let's try manual save/load using get_state/load_state if exposed, or torch.save
    
    state = model.get_state()
    torch.save(state, save_path)
    
    new_model = AIM(epsilon=1.0)
    new_model.load_state(save_path)
    
    assert new_model.model_loaded is True
    # Can we sample from loaded model?
    samples = new_model.sample(5, return_dataframe=True)
    assert len(samples) == 5
