import numpy as np
import pandas as pd
import pytest
import torch

from stg.TabPFGen.tabpfgen_synthesizer import TabPFGenSynthesizer

pytestmark = pytest.mark.gpu


# ------------------------------------------------------------------
# Existing tests (preserved)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# New tests
# ------------------------------------------------------------------
def test_tabpfgen_numeric_only():
    np.random.seed(42)
    df = pd.DataFrame({
        "x": np.random.randn(80),
        "y": np.random.rand(80) * 10,
        "target": np.random.randn(80),
    })
    model = TabPFGenSynthesizer(n_sgld_steps=2, random_state=42)
    model.fit(df)
    samples = model.sample(8, return_dataframe=True)
    assert samples.shape == (8, 3)
    assert pd.api.types.is_numeric_dtype(samples["x"])
    assert pd.api.types.is_numeric_dtype(samples["y"])


def test_tabpfgen_explicit_target(sample_data):
    model = TabPFGenSynthesizer(
        n_sgld_steps=2, target_column="target", random_state=42
    )
    model.fit(sample_data)
    samples = model.sample(5, return_dataframe=True)
    assert samples.shape == (5, sample_data.shape[1])
    assert "target" in samples.columns


def test_tabpfgen_regression_target():
    np.random.seed(42)
    df = pd.DataFrame({
        "feat1": np.random.randn(80),
        "feat2": np.random.rand(80),
        "value": np.random.randn(80) * 100,  # continuous target
    })
    model = TabPFGenSynthesizer(
        n_sgld_steps=2, target_column="value", random_state=42
    )
    model.fit(df)
    samples = model.sample(5, return_dataframe=True)
    assert samples.shape == (5, 3)
    assert pd.api.types.is_numeric_dtype(samples["value"])


def test_tabpfgen_reproducibility(sample_data):
    """Two runs with same seed should produce similar (not necessarily identical) output.

    SGLD injects noise at each step, so exact reproducibility depends on torch
    internals. We check that results are reasonably close (decimal=1) rather
    than bit-exact.
    """
    model1 = TabPFGenSynthesizer(n_sgld_steps=2, random_state=99)
    model1.fit(sample_data)
    s1 = model1.sample(5, return_dataframe=True)

    model2 = TabPFGenSynthesizer(n_sgld_steps=2, random_state=99)
    model2.fit(sample_data)
    s2 = model2.sample(5, return_dataframe=True)

    np.testing.assert_array_almost_equal(
        s1["feature1"].values, s2["feature1"].values, decimal=1
    )


def test_tabpfgen_custom_sgld_params(sample_data):
    model = TabPFGenSynthesizer(
        n_sgld_steps=5,
        sgld_step_size=0.05,
        sgld_noise_scale=0.02,
        random_state=42,
    )
    model.fit(sample_data)
    samples = model.sample(5, return_dataframe=True)
    assert samples.shape == (5, sample_data.shape[1])


def test_tabpfgen_factory(sample_data):
    from stg.tableSynthesizer import TableSynthesizer, DEFAULT_MODELS

    if "TabPFGen" not in DEFAULT_MODELS:
        pytest.skip("TabPFGen not registered in factory")

    ts = TableSynthesizer("TabPFGen", config={"n_sgld_steps": 2, "random_state": 42})
    ts.fit(sample_data)
    samples = ts.sample(5, return_dataframe=True)
    assert samples.shape == (5, sample_data.shape[1])


def test_tabpfgen_single_feature():
    np.random.seed(42)
    df = pd.DataFrame({
        "feat": np.random.randn(80),
        "target": np.random.choice(["A", "B"], 80),
    })
    model = TabPFGenSynthesizer(n_sgld_steps=2, random_state=42)
    model.fit(df)
    samples = model.sample(5, return_dataframe=True)
    assert samples.shape == (5, 2)


def test_tabpfgen_large_sample(sample_data):
    model = TabPFGenSynthesizer(n_sgld_steps=2, random_state=42)
    model.fit(sample_data)
    # Generate more rows than training data
    samples = model.sample(200, return_dataframe=True)
    assert samples.shape == (200, sample_data.shape[1])
