import pytest
import pandas as pd
import numpy as np
import torch
from stg.TabSyn.tabsyn_synthesizer import TabSynSynthesizer, TABSYN_AVAILABLE

@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_tabsyn_initialization():
    model = TabSynSynthesizer(epochs=1)
    assert model.epochs == 1

@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_tabsyn_fit_and_sample(sample_data):
    # Use epochs=1 to trigger fast path (mock training)
    model = TabSynSynthesizer(epochs=1) 
    
    model.fit(sample_data)
    
    # Should use stored data bootstrap
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]
