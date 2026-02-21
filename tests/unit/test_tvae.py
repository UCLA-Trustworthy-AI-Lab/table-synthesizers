import pytest
import pandas as pd
import numpy as np
import torch
from stg.TVAE.TVAE import TVAE

pytestmark = pytest.mark.gpu

def test_tvae_initialization():
    model = TVAE(epochs=1)
    assert model._epochs == 1

def test_tvae_fit_and_sample(sample_data):
    model = TVAE(epochs=1, batch_size=10)
    model.fit(sample_data)
    
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]
