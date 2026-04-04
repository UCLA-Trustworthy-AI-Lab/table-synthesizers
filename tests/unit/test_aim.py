import pytest
import pandas as pd
import numpy as np
import torch
import shutil
import os
import importlib
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

def test_aim_fit_and_sample_mixed_data(sample_data):
    """Mixed-type sampling should decode back to the original column schema."""
    model = AIM(epsilon=1.0, epochs=1, continuous_binning="quantile", continuous_bin_count=16)

    model.fit(sample_data)
    samples = model.sample(10, return_dataframe=True)

    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (10, sample_data.shape[1])
    assert list(samples.columns) == list(sample_data.columns)
    assert set(samples["target"].unique()).issubset(set(sample_data["target"].unique()))

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

def test_aim_save_load_preserves_decoder_metadata(sample_data, tmp_path):
    """Checkpoints should retain enough metadata to decode mixed-type samples."""
    model = AIM(epsilon=1.0, epochs=1, continuous_bin_count=8)
    model.fit(sample_data)

    save_path = tmp_path / "aim_mixed_checkpoint.pt"
    torch.save(model.get_state(), save_path)

    restored_model = AIM(epsilon=1.0)
    restored_model.load_state(save_path)

    restored_samples = restored_model.sample(6, return_dataframe=True)

    assert isinstance(restored_samples, pd.DataFrame)
    assert restored_samples.shape == (6, sample_data.shape[1])
    assert list(restored_samples.columns) == list(sample_data.columns)

def test_aim_auto_backend_prefers_snsynth_when_available(sample_categorical_data, monkeypatch):
    """Auto backend should route to snsynth when that backend is available."""
    aim_module = importlib.import_module("stg.AIM.AIM")

    class DummySNSynth:
        def __init__(self):
            self.fitted_df = None

        @classmethod
        def create(cls, *args, **kwargs):
            return cls()

        def fit(self, df, preprocessor_eps=0.0):
            self.fitted_df = df.copy()

        def sample(self, n):
            return pd.concat([self.fitted_df.iloc[[0]].copy() for _ in range(n)], ignore_index=True)

    monkeypatch.setattr(aim_module, "SNSYNTH_AVAILABLE", True)
    monkeypatch.setattr(aim_module, "SNSynthesizer", DummySNSynth)

    model = aim_module.AIM(epsilon=1.0, epochs=1, backend="auto")
    assert model.backend == "snsynth"

    model.fit(sample_categorical_data)
    samples = model.sample(4, return_dataframe=True)

    assert isinstance(model.model, DummySNSynth)
    assert isinstance(samples, pd.DataFrame)
    assert list(samples.columns) == list(sample_categorical_data.columns)
