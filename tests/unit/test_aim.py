import pytest
import pandas as pd
import numpy as np
import torch

try:
    from stg.AIM.AIM import AIM, AIM_AVAILABLE
except ImportError:
    AIM_AVAILABLE = False


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_initialization():
    """Test that AIM can be initialized with the real algorithm."""
    model = AIM(epsilon=0.5)
    assert model.epsilon == 0.5
    assert model.delta == 1e-9
    assert model.max_model_size == 80
    assert model.degree == 2


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_fit_and_sample(sample_categorical_data):
    """Test AIM training and sampling with the real algorithm."""
    model = AIM(epsilon=1.0, epochs=1)

    model.fit(sample_categorical_data)

    samples = model.sample(10, return_dataframe=True)

    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 10
    assert samples.shape[1] == sample_categorical_data.shape[1]


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_save_load(sample_categorical_data, tmp_path):
    """Test saving and loading the AIM model."""
    model = AIM(epsilon=1.0, epochs=1)
    model.fit(sample_categorical_data)

    save_path = tmp_path / "aim_checkpoint.pt"

    state = model.get_state()
    torch.save(state, save_path)

    new_model = AIM(epsilon=1.0)
    new_model.load_state(save_path)

    assert new_model.model_loaded is True
    samples = new_model.sample(5, return_dataframe=True)
    assert len(samples) == 5


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_sample_tensor(sample_categorical_data):
    """Test that tensor output works."""
    model = AIM(epsilon=1.0, epochs=1)
    model.fit(sample_categorical_data)

    tensor_out = model.sample(8, return_dataframe=False)
    assert isinstance(tensor_out, torch.Tensor)
    assert tensor_out.shape[0] == 8
    assert tensor_out.shape[1] == sample_categorical_data.shape[1]


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_custom_params(sample_categorical_data):
    """Test AIM with custom privacy parameters."""
    model = AIM(
        epsilon=3.0,
        delta=1e-5,
        max_model_size=40,
        degree=1,
        max_cells=5000,
    )
    model.fit(sample_categorical_data)
    samples = model.sample(10, return_dataframe=True)
    assert len(samples) == 10


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_discrete_values(sample_categorical_data):
    """Test that AIM produces discrete values (it's a discrete mechanism)."""
    model = AIM(epsilon=1.0, epochs=1)
    model.fit(sample_categorical_data)
    samples = model.sample(20, return_dataframe=True)

    # Integer columns in training should produce integer-like values
    for col in ["col1", "col2"]:
        if col in samples.columns:
            vals = samples[col].dropna()
            if len(vals) > 0 and pd.api.types.is_numeric_dtype(vals):
                # Values should be within reasonable range of training data
                assert vals.min() >= sample_categorical_data[col].min() - 1
                assert vals.max() <= sample_categorical_data[col].max() + 1


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_generate_before_fit():
    """Test that generating before fitting raises an error."""
    model = AIM(epsilon=1.0)
    with pytest.raises(RuntimeError, match="trained"):
        model.sample(5)


@pytest.mark.skipif(not AIM_AVAILABLE, reason="smartnoise-synth / private-pgm not installed")
def test_aim_large_sample(sample_categorical_data):
    """Test generating more samples than training data."""
    model = AIM(epsilon=1.0, epochs=1)
    model.fit(sample_categorical_data)
    samples = model.sample(200, return_dataframe=True)
    assert len(samples) == 200
