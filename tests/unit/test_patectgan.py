import pytest
import pandas as pd
import numpy as np
import torch
from stg.PATECTGAN.patectgan import PATECTGAN

pytestmark = pytest.mark.gpu

def test_patectgan_initialization():
    model = PATECTGAN(epochs=1)
    assert model._epochs == 1

def test_patectgan_fit_and_sample(sample_data):
    # Set episodes to minimal for speed
    model = PATECTGAN(epochs=1, batch_size=10, verbose=False, teacher_iters=1, student_iters=1)
    model.fit(sample_data)
    
    samples = model.sample(10, return_dataframe=True)
    
    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]
