import pytest
import pandas as pd
import numpy as np
import torch
from stg.LTM_VAE import LTMVAESynthesizer, LTM_VAE_AVAILABLE

@pytest.mark.skipif(not LTM_VAE_AVAILABLE, reason="LTM-VAE not installed")
def test_ltm_vae_initialization():
    model = LTMVAESynthesizer(num_epochs=1)
    assert model.num_epochs == 1

@pytest.mark.skipif(not LTM_VAE_AVAILABLE, reason="LTM-VAE not installed")
def test_ltm_vae_fit_and_sample(sample_data):
    model = LTMVAESynthesizer(num_epochs=1, config_task='quick_test')
    
    model.fit(sample_data)
    
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    # LTM typically returns same columns
    assert samples.shape[1] == sample_data.shape[1]
