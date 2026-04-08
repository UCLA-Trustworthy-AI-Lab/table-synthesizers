import numpy as np
import pandas as pd
import pytest
import torch

from stg.TabDiff.tabdiff_synthesizer import TabDiffSynthesizer

pytestmark = pytest.mark.gpu


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------
def test_tabdiff_initialization():
    model = TabDiffSynthesizer()
    assert model.stored_data is None


def test_tabdiff_default_hyperparams():
    model = TabDiffSynthesizer()
    assert model.num_diffusion_steps == 50
    assert model._epochs == 8000
    assert model._lr == 1e-3
    assert model.hidden_dims == (256, 256, 256)


def test_tabdiff_custom_hyperparams():
    model = TabDiffSynthesizer(
        num_diffusion_steps=10,
        epochs=3,
        hidden_dims=(64,),
        learning_rate=5e-4,
    )
    assert model.num_diffusion_steps == 10
    assert model._epochs == 3
    assert model.hidden_dims == (64,)
    assert model._lr == 5e-4


def test_tabdiff_deprecated_steps_alias():
    with pytest.warns(DeprecationWarning, match="deprecated alias"):
        model = TabDiffSynthesizer(steps=3)

    assert model._epochs == 3


def test_tabdiff_rejects_conflicting_epochs_and_steps():
    with pytest.raises(ValueError, match="either `epochs` or the deprecated `steps` alias"):
        TabDiffSynthesizer(epochs=4, steps=3)


# ------------------------------------------------------------------
# Fit + Sample (all use epochs=1 fast path)
# ------------------------------------------------------------------
def test_tabdiff_fit_and_sample(sample_data):
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(sample_data)

    samples = model.sample(12, return_dataframe=True)
    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (12, sample_data.shape[1])
    assert list(samples.columns) == list(sample_data.columns)


def test_tabdiff_edit(sample_data):
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(sample_data)

    row = sample_data.iloc[[0]].copy()
    row["feature1"] = np.nan

    edited = model.edit(row, intervention={"target": "A"}, n_samples=5)
    assert edited.shape == (5, sample_data.shape[1])
    assert (edited["target"] == "A").all()


def test_tabdiff_numeric_only():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(80),
        "b": np.random.rand(80) * 100,
    })
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(df)
    samples = model.sample(10, return_dataframe=True)
    assert samples.shape == (10, 2)
    assert list(samples.columns) == ["a", "b"]
    assert pd.api.types.is_numeric_dtype(samples["a"])


def test_tabdiff_categorical_only():
    np.random.seed(42)
    df = pd.DataFrame({
        "color": np.random.choice(["red", "blue", "green"], 80),
        "size": np.random.choice(["S", "M", "L"], 80),
    })
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(df)
    samples = model.sample(10, return_dataframe=True)
    assert samples.shape == (10, 2)
    assert set(samples["color"].unique()).issubset({"red", "blue", "green"})
    assert set(samples["size"].unique()).issubset({"S", "M", "L"})


def test_tabdiff_dtypes_preserved(sample_data):
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(sample_data)
    samples = model.sample(10, return_dataframe=True)

    assert pd.api.types.is_numeric_dtype(samples["feature1"])
    assert pd.api.types.is_numeric_dtype(samples["feature2"])
    assert all(isinstance(v, str) for v in samples["target"])


def test_tabdiff_generate_returns_tensor(sample_data):
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(sample_data)
    tensor_out = model.sample(8, return_dataframe=False)
    assert isinstance(tensor_out, torch.Tensor)
    assert tensor_out.shape[0] == 8


def test_tabdiff_reproducibility(sample_data):
    model1 = TabDiffSynthesizer(epochs=1, random_state=123)
    model1.fit(sample_data)
    s1 = model1.sample(5, return_dataframe=True)

    model2 = TabDiffSynthesizer(epochs=1, random_state=123)
    model2.fit(sample_data)
    s2 = model2.sample(5, return_dataframe=True)

    np.testing.assert_array_almost_equal(
        s1["feature1"].values, s2["feature1"].values, decimal=4
    )


def test_tabdiff_get_state_load_state(sample_data):
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(sample_data)

    state = model.get_state()
    assert state is not None

    model2 = TabDiffSynthesizer()
    model2.load_state(state)

    samples = model2.sample(5, return_dataframe=True)
    assert samples.shape == (5, sample_data.shape[1])


def test_tabdiff_single_column():
    np.random.seed(42)
    df = pd.DataFrame({"x": np.random.randn(50)})
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(df)
    samples = model.sample(5, return_dataframe=True)
    assert samples.shape == (5, 1)
    assert list(samples.columns) == ["x"]


def test_tabdiff_condition_dict(sample_data):
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(sample_data)
    samples = model.generate(5, condition={"target": "B"})
    decoded = model.decode_samples(samples)
    assert len(decoded) == 5


def test_tabdiff_factory(sample_data):
    from stg.tableSynthesizer import TableSynthesizer, DEFAULT_MODELS

    if "TabDiff" not in DEFAULT_MODELS:
        pytest.skip("TabDiff not registered in factory")

    ts = TableSynthesizer("TabDiff", config={"epochs": 1, "random_state": 42})
    ts.fit(sample_data)
    samples = ts.sample(5, return_dataframe=True)
    assert samples.shape == (5, sample_data.shape[1])


def test_tabdiff_rejects_dataloader(sample_data):
    model = TabDiffSynthesizer()
    tensor_data = torch.randn(100, 3)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        model.fit(loader)


def test_tabdiff_large_sample(sample_data):
    model = TabDiffSynthesizer(epochs=1, random_state=42)
    model.fit(sample_data)
    samples = model.sample(200, return_dataframe=True)
    assert samples.shape == (200, sample_data.shape[1])
