"""Integration tests for model training on different devices.

Tests CTGAN fit+sample on CPU (always) and CUDA (when available).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from stg.gpu_utils import detect_best_device, is_gpu_available
from stg.tableSynthesizer import TableSynthesizer

HAS_CUDA = torch.cuda.is_available()
HAS_GPU = is_gpu_available()


@pytest.fixture
def small_mixed_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 100),
            "income": np.random.randint(20000, 150000, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )


def test_ctgan_trains_on_cpu(small_mixed_data):
    config = {"epochs": 1, "batch_size": 50, "embedding_dim": 32}
    synth = TableSynthesizer("CTGAN", config=config)
    synth.model.set_device("cpu")
    synth.fit(small_mixed_data)

    out = synth.sample(n=10, return_dataframe=True)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 10
    assert set(out.columns) == set(small_mixed_data.columns)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
@pytest.mark.cuda
def test_ctgan_trains_on_cuda(small_mixed_data):
    config = {"epochs": 1, "batch_size": 50, "embedding_dim": 32}
    synth = TableSynthesizer("CTGAN", config=config)
    synth.model.set_device("cuda")
    synth.fit(small_mixed_data)

    # Verify model parameters are on CUDA
    gen_params = list(synth.model._generator.parameters())
    assert gen_params, "Generator has no parameters"
    assert gen_params[0].is_cuda, "Generator params not on CUDA"

    out = synth.sample(n=10, return_dataframe=True)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 10


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
@pytest.mark.cuda
def test_cuda_memory_allocated_after_training(small_mixed_data):
    torch.cuda.reset_peak_memory_stats()
    config = {"epochs": 1, "batch_size": 50, "embedding_dim": 32}
    synth = TableSynthesizer("CTGAN", config=config)
    synth.model.set_device("cuda")
    synth.fit(small_mixed_data)

    assert torch.cuda.max_memory_allocated(0) > 0


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
@pytest.mark.gpu
def test_auto_device_training(small_mixed_data):
    device = detect_best_device()
    config = {"epochs": 1, "batch_size": 50, "embedding_dim": 32}
    synth = TableSynthesizer("CTGAN", config=config)
    synth.model.set_device(str(device))
    synth.fit(small_mixed_data)

    out = synth.sample(n=10, return_dataframe=True)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 10
