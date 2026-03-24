import pandas as pd
import pytest

try:
    from stg.GaussianCopula.gaussian_copula_synthesizer import (
        SDV_AVAILABLE,
        GaussianCopulaSynthesizer,
    )
except ImportError:
    SDV_AVAILABLE = False
    GaussianCopulaSynthesizer = None


@pytest.mark.skipif(not SDV_AVAILABLE, reason="sdv not installed")
def test_gaussian_copula_initialization():
    model = GaussianCopulaSynthesizer()
    assert model.model is None


@pytest.mark.skipif(not SDV_AVAILABLE, reason="sdv not installed")
def test_gaussian_copula_fit_and_sample(sample_data):
    model = GaussianCopulaSynthesizer()
    model.fit(sample_data)

    samples = model.sample(10, return_dataframe=True)
    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (10, sample_data.shape[1])
    assert list(samples.columns) == list(sample_data.columns)


@pytest.mark.skipif(not SDV_AVAILABLE, reason="sdv not installed")
def test_gaussian_copula_edit(sample_data):
    model = GaussianCopulaSynthesizer()
    model.fit(sample_data)
    edited = model.edit(sample_data.iloc[[0]], intervention={"target": "A"}, n_samples=5)
    assert edited.shape == (5, sample_data.shape[1])
    assert (edited["target"] == "A").all()
