import pytest
import pandas as pd
import numpy as np
import torch
from stg.GREAT.great_synthesizer import GREATSynthesizer, SYNTHCITY_AVAILABLE

@pytest.mark.skipif(not SYNTHCITY_AVAILABLE, reason="synthcity not installed")
def test_great_initialization():
    model = GREATSynthesizer()
    assert model.model is None

@pytest.mark.skipif(not SYNTHCITY_AVAILABLE, reason="synthcity not installed")
def test_great_fit_and_sample(sample_data):
    model = GREATSynthesizer()
    model.fit(sample_data)
    samples = model.sample(10, return_dataframe=True)
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]
