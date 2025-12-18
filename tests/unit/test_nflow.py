import pytest
import pandas as pd
import numpy as np
import torch
from stg.NFlow.nflow_synthesizer import NFlowSynthesizer, SYNTHCITY_AVAILABLE

@pytest.mark.skipif(not SYNTHCITY_AVAILABLE, reason="synthcity not installed")
def test_nflow_initialization():
    model = NFlowSynthesizer()
    assert model.model is None

@pytest.mark.skipif(not SYNTHCITY_AVAILABLE, reason="synthcity not installed")
def test_nflow_fit_and_sample(sample_data):
    model = NFlowSynthesizer()
    model.fit(sample_data)
    samples = model.sample(10, return_dataframe=True)
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]
