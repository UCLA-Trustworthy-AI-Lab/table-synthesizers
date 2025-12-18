import pytest
import pandas as pd
import numpy as np
import torch
from stg.SMOTE.smote_synthesizer import SMOTESynthesizer

def test_smote_initialization():
    model = SMOTESynthesizer()
    assert model.stored_data is None

def test_smote_fit_and_sample(sample_data):
    model = SMOTESynthesizer(k_neighbors=2) # low k for small sample data
    model.fit(sample_data)
    
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]
