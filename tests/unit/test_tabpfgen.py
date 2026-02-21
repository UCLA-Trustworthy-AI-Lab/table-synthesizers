import numpy as np
import pandas as pd
import pytest

from stg.TabPFGen.tabpfgen_synthesizer import TabPFGenSynthesizer

pytestmark = pytest.mark.gpu


def test_tabpfgen_initialization():
    model = TabPFGenSynthesizer(n_sgld_steps=2)
    assert model.n_sgld_steps == 2


def test_tabpfgen_fit_and_sample(sample_data):
    model = TabPFGenSynthesizer(n_sgld_steps=3, random_state=42)
    model.fit(sample_data)

    samples = model.sample(10, return_dataframe=True)
    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (10, sample_data.shape[1])
    assert list(samples.columns) == list(sample_data.columns)


def test_tabpfgen_edit(sample_data):
    model = TabPFGenSynthesizer(n_sgld_steps=2, random_state=42)
    model.fit(sample_data)

    row = sample_data.iloc[[0]].copy()
    row["feature2"] = np.nan
    edited = model.edit(row, intervention={"target": "B"}, n_samples=4)

    assert edited.shape == (4, sample_data.shape[1])
    assert (edited["target"] == "B").all()


def test_tabpfgen_tensor_output(sample_data):
    model = TabPFGenSynthesizer(n_sgld_steps=2, random_state=42)
    model.fit(sample_data)
    tensor_samples = model.sample(6, return_dataframe=False)

    assert tensor_samples.shape[0] == 6
    assert tensor_samples.shape[1] == sample_data.shape[1]
