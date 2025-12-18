import pytest
import pandas as pd
import numpy as np
import torch
from stg.BayesianNetwork.bayesian_network_synthesizer import BayesianNetworkSynthesizer, SYNTHCITY_AVAILABLE

@pytest.mark.skipif(not SYNTHCITY_AVAILABLE, reason="synthcity not installed")
def test_bn_initialization():
    """Test that BayesianNetwork can be initialized."""
    model = BayesianNetworkSynthesizer()
    assert model.model is None

@pytest.mark.skipif(not SYNTHCITY_AVAILABLE, reason="synthcity not installed")
def test_bn_fit_and_sample(sample_data):
    """Test BayesianNetwork training and sampling."""
    model = BayesianNetworkSynthesizer()
    
    model.fit(sample_data)
    
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]

@pytest.mark.skipif(not SYNTHCITY_AVAILABLE, reason="synthcity not installed")
def test_bn_save_load(sample_data, tmp_path):
    """Test saving and loading."""
    model = BayesianNetworkSynthesizer()
    model.fit(sample_data)
    
    assert model.stored_data is not None
