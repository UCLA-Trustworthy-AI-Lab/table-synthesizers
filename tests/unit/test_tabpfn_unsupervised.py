import numpy as np
import pandas as pd
import pytest
import torch

try:
    from stg.TabPFNUnsupervised.tabpfn_unsupervised_synthesizer import (
        TabPFNUnsupervisedSynthesizer,
        TABPFN_EXTENSIONS_AVAILABLE,
    )
except ImportError:
    TABPFN_EXTENSIONS_AVAILABLE = False

pytestmark = pytest.mark.gpu


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------
def test_tabpfn_unsupervised_availability():
    """Verify that the availability flag is a boolean."""
    assert isinstance(TABPFN_EXTENSIONS_AVAILABLE, bool)


# ------------------------------------------------------------------
# Tests that require tabpfn-extensions
# ------------------------------------------------------------------
@pytest.mark.skipif(
    not TABPFN_EXTENSIONS_AVAILABLE,
    reason="tabpfn / tabpfn-extensions not installed",
)
class TestTabPFNUnsupervisedWithDeps:

    def test_init(self):
        model = TabPFNUnsupervisedSynthesizer()
        assert model.t == 1.0
        assert model.n_permutations == 3
        assert model.stored_data is None

    def test_custom_params(self):
        model = TabPFNUnsupervisedSynthesizer(t=0.5, n_permutations=5)
        assert model.t == 0.5
        assert model.n_permutations == 5

    def test_fit_and_sample(self, sample_data):
        model = TabPFNUnsupervisedSynthesizer(t=1.0, n_permutations=1)
        model.fit(sample_data)

        samples = model.sample(10, return_dataframe=True)
        assert isinstance(samples, pd.DataFrame)
        assert samples.shape == (10, sample_data.shape[1])
        assert list(samples.columns) == list(sample_data.columns)

    def test_numeric_only(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(80),
            "b": np.random.rand(80) * 100,
        })
        model = TabPFNUnsupervisedSynthesizer(t=1.0, n_permutations=1)
        model.fit(df)
        samples = model.sample(8, return_dataframe=True)
        assert samples.shape == (8, 2)
        assert pd.api.types.is_numeric_dtype(samples["a"])

    def test_dtypes_preserved(self, sample_data):
        model = TabPFNUnsupervisedSynthesizer(t=1.0, n_permutations=1)
        model.fit(sample_data)
        samples = model.sample(5, return_dataframe=True)

        assert pd.api.types.is_numeric_dtype(samples["feature1"])
        assert pd.api.types.is_numeric_dtype(samples["feature2"])
        # Categorical column restored to string
        assert all(isinstance(v, str) for v in samples["target"])

    def test_sample_tensor(self, sample_data):
        model = TabPFNUnsupervisedSynthesizer(t=1.0, n_permutations=1)
        model.fit(sample_data)
        tensor_out = model.sample(6, return_dataframe=False)
        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.shape[0] == 6

    def test_single_column(self):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.randn(50)})
        model = TabPFNUnsupervisedSynthesizer(t=1.0, n_permutations=1)
        model.fit(df)
        samples = model.sample(5, return_dataframe=True)
        assert samples.shape == (5, 1)

    def test_factory(self, sample_data):
        from stg.tableSynthesizer import DEFAULT_MODELS

        if "TabPFNUnsupervised" not in DEFAULT_MODELS:
            pytest.skip("TabPFNUnsupervised not registered in factory")

        from stg.tableSynthesizer import TableSynthesizer

        ts = TableSynthesizer(
            "TabPFNUnsupervised", config={"t": 1.0, "n_permutations": 1}
        )
        ts.fit(sample_data)
        samples = ts.sample(5, return_dataframe=True)
        assert samples.shape == (5, sample_data.shape[1])

    def test_rejects_dataloader(self):
        model = TabPFNUnsupervisedSynthesizer()
        tensor_data = torch.randn(100, 3)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        with pytest.raises(ValueError, match="DataFrame"):
            model.fit(loader)

    def test_decode_samples(self, sample_data):
        model = TabPFNUnsupervisedSynthesizer(t=1.0, n_permutations=1)
        model.fit(sample_data)
        tensor_out = model.generate(5)
        decoded = model.decode_samples(tensor_out)
        assert isinstance(decoded, pd.DataFrame)
        assert decoded.shape[0] == 5


# ------------------------------------------------------------------
# Test that runs when deps are NOT available
# ------------------------------------------------------------------
@pytest.mark.skipif(
    TABPFN_EXTENSIONS_AVAILABLE,
    reason="Test only runs when tabpfn-extensions is NOT installed",
)
def test_tabpfn_unsupervised_raises_without_deps():
    from stg.TabPFNUnsupervised.tabpfn_unsupervised_synthesizer import (
        TabPFNUnsupervisedSynthesizer,
    )

    with pytest.raises(ImportError, match="tabpfn"):
        TabPFNUnsupervisedSynthesizer()
