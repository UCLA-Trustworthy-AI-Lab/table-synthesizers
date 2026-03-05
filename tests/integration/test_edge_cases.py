"""
Edge-case test suite for table-synthesizers.

Covers corner-case data scenarios that every model should handle gracefully:
  - numerical-only data
  - categorical-only data
  - unicode column names and values
  - binary data
  - nested JSON / stringified JSON columns
  - single-column DataFrame
  - single-row DataFrame
  - constant columns (zero variance)
  - high-cardinality categorical
  - mixed NaN patterns
  - empty-string categorical values
  - boolean columns
  - wide (many columns) DataFrame

Models are split into three tiers with pytest markers:
  - edge_core:      lightweight / always-available models
  - edge_synthcity: synthcity-backend models (ARF, BayesianNetwork, GREAT, NFlow)
  - edge_gpu:       GPU-heavy models (CTGAN, PATECTGAN, TabDDPM, AutoDiff, TabSyn)

Run specific tiers:
  pytest tests/integration/test_edge_cases.py -m edge_core
  pytest tests/integration/test_edge_cases.py -m edge_synthcity
  pytest tests/integration/test_edge_cases.py -m edge_gpu
  pytest tests/integration/test_edge_cases.py          # all tiers
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from stg.tableSynthesizer import TableSynthesizer, DEFAULT_MODELS

# ---------------------------------------------------------------------------
# Models to test -- split by tier
# ---------------------------------------------------------------------------

# Always-available lightweight models (no optional deps)
CORE_MODELS = ["Identity", "CART", "DPCART"]

# Models that require torch but are always importable when torch exists
TORCH_MODELS = [
    m for m in ["TabDiff", "TabPFGen", "TVAE"]
    if m in DEFAULT_MODELS
]

# Optional models gated by availability flags
OPTIONAL_MODELS = [
    m for m in ["GaussianCopula"]
    if m in DEFAULT_MODELS
]

# Synthcity-backend models (require synthcity package)
SYNTHCITY_MODELS = [
    m for m in ["ARF", "BayesianNetwork", "GREAT", "NFlow"]
    if m in DEFAULT_MODELS
]

# GPU-heavy models (GAN, diffusion, VAE -- typically need GPU for reasonable speed)
GPU_MODELS = [
    m for m in ["CTGAN", "PATECTGAN", "TabDDPM", "AutoDiff", "TabSyn"]
    if m in DEFAULT_MODELS
]

# Union of core lightweight models (backwards compat)
EDGE_CASE_MODELS = CORE_MODELS + TORCH_MODELS + OPTIONAL_MODELS

# Full set across all tiers
ALL_EDGE_MODELS = EDGE_CASE_MODELS + SYNTHCITY_MODELS + GPU_MODELS

# Models that accept random_state for reproducibility testing
# Note: TabPFGen uses SGLD with torch stochastic ops, so determinism is not guaranteed
SEEDED_MODELS = [
    m for m in ["TabDiff", "CART", "DPCART"]
    if m in DEFAULT_MODELS
]

# Models that require at least 2 columns (need target + features)
MULTI_COLUMN_ONLY = {"DPCART", "TabPFGen"}

# Models that cannot handle NaN values natively
# NFlow: synthcity's NFlow uses BayesianGaussianMixture for encoding which rejects NaN
NO_NAN_SUPPORT = {"CART", "DPCART", "TVAE", "CTGAN", "PATECTGAN", "AutoDiff", "TabSyn", "NFlow"}

N_SAMPLES = 10  # keep small for speed

# Apply edge_case marker to all tests in this module
pytestmark = pytest.mark.edge_case

# Fast CI configs for slow models (overridden only when caller doesn't specify)
_FAST_CI_CONFIG = {
    "GREAT": {"n_iter": 1},   # default 100 LLM epochs is too slow for CI
    "NFlow": {"n_iter": 5},   # default 1000 flow iterations is too slow for CI
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_sample(model_name: str, df: pd.DataFrame, config: dict | None = None):
    """Fit a model and return sampled DataFrame."""
    base = _FAST_CI_CONFIG.get(model_name, {})
    config = {**base, **(config or {})}
    synth = TableSynthesizer(model_name, config)
    synth.fit(df)
    return synth.sample(n=N_SAMPLES, return_dataframe=True)


def _assert_basic(result: pd.DataFrame, expected_cols: set, n: int = N_SAMPLES):
    """Common assertions on a sampled DataFrame."""
    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"
    assert result.shape[0] == n, f"Expected {n} rows, got {result.shape[0]}"
    assert set(result.columns) == expected_cols, (
        f"Column mismatch: expected {expected_cols}, got {set(result.columns)}"
    )


def _models_for(models=None):
    """Return a list of model names filtered by availability."""
    models = models or EDGE_CASE_MODELS
    return [m for m in models if m in DEFAULT_MODELS]


def _parametrize_all():
    """Return all models as pytest.param with tier markers for parametrize.

    Each model gets its tier marker so tests can be filtered:
      pytest -m edge_core         # core/lightweight only
      pytest -m edge_synthcity    # synthcity models only
      pytest -m edge_gpu          # GPU-heavy models only
    """
    params = []
    for m in _models_for(EDGE_CASE_MODELS):
        params.append(pytest.param(m, marks=pytest.mark.edge_core))
    for m in _models_for(SYNTHCITY_MODELS):
        params.append(pytest.param(m, marks=pytest.mark.edge_synthcity))
    for m in _models_for(GPU_MODELS):
        params.append(pytest.param(m, marks=pytest.mark.edge_gpu))
    return params


def _parametrize_seeded():
    """Return seeded models with tier markers for reproducibility tests."""
    core_seeded = [m for m in SEEDED_MODELS if m in EDGE_CASE_MODELS]
    params = []
    for m in core_seeded:
        if m in DEFAULT_MODELS:
            params.append(pytest.param(m, marks=pytest.mark.edge_core))
    # Synthcity models with deterministic seeding
    for m in _models_for(["BayesianNetwork", "ARF"]):
        params.append(pytest.param(m, marks=pytest.mark.edge_synthcity))
    return params


# ---------------------------------------------------------------------------
# 1. Numerical-only data
# ---------------------------------------------------------------------------

class TestNumericalOnly:
    """All columns are numeric (float64 / int64)."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "price": np.random.uniform(10, 1000, 100),
            "quantity": np.random.randint(1, 50, 100),
            "rating": np.random.normal(3.5, 0.8, 100),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_numerical_only_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))
        # All output columns should still be numeric
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), (
                f"Column {col} is {result[col].dtype}, expected numeric"
            )


# ---------------------------------------------------------------------------
# 2. Categorical-only data
# ---------------------------------------------------------------------------

class TestCategoricalOnly:
    """All columns are categorical / object dtype."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "color": np.random.choice(["red", "green", "blue"], 100),
            "size": np.random.choice(["S", "M", "L", "XL"], 100),
            "region": np.random.choice(["US", "EU", "APAC"], 100),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_categorical_only_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 3. Unicode column names and values
# ---------------------------------------------------------------------------

class TestUnicode:
    """Column names and values contain non-ASCII / CJK characters."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "\u4ef7\u683c": np.random.uniform(10, 500, 80),          # 价格 (price)
            "\u30ab\u30c6\u30b4\u30ea": np.random.choice(            # カテゴリ (category)
                ["\u98df\u54c1", "\u98f2\u6599", "\u65e5\u7528\u54c1"], 80  # 食品, 飲料, 日用品
            ),
            "sch\u00f6n_\u00fc": np.random.choice(                   # schön_ü (German)
                ["gro\u00df", "klein"], 80                           # groß, klein
            ),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_unicode_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 4. Binary data
# ---------------------------------------------------------------------------

class TestBinaryData:
    """Columns with exactly two distinct values (bool, 0/1, yes/no)."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "flag_bool": np.random.choice([True, False], n),
            "flag_int": np.random.choice([0, 1], n),
            "flag_str": np.random.choice(["yes", "no"], n),
            "value": np.random.randn(n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_binary_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 5. Nested JSON / stringified JSON column
# ---------------------------------------------------------------------------

class TestNestedJSON:
    """One column contains serialised JSON strings (common in log/event data)."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 80
        jsons = [
            json.dumps({"key": f"v{i}", "nested": {"a": int(np.random.randint(0, 10))}})
            for i in range(n)
        ]
        return pd.DataFrame({
            "id": np.arange(n),
            "payload": jsons,
            "score": np.random.uniform(0, 1, n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_json_column_fit_sample(self, model, df):
        """Models should treat JSON strings as opaque categorical values."""
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 6. Single-column DataFrame
# ---------------------------------------------------------------------------

class TestSingleColumn:
    """DataFrame with only one column."""

    @pytest.fixture
    def df_num(self):
        np.random.seed(42)
        return pd.DataFrame({"value": np.random.randn(100)})

    @pytest.fixture
    def df_cat(self):
        np.random.seed(42)
        return pd.DataFrame({"label": np.random.choice(["A", "B", "C"], 100)})

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_single_numeric_column(self, model, df_num):
        if model in MULTI_COLUMN_ONLY:
            pytest.skip(f"{model} requires at least 2 columns (target + features)")
        result = _fit_sample(model, df_num)
        _assert_basic(result, {"value"})

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_single_categorical_column(self, model, df_cat):
        if model in MULTI_COLUMN_ONLY:
            pytest.skip(f"{model} requires at least 2 columns (target + features)")
        result = _fit_sample(model, df_cat)
        _assert_basic(result, {"label"})


# ---------------------------------------------------------------------------
# 7. Constant columns (zero variance)
# ---------------------------------------------------------------------------

class TestConstantColumns:
    """Columns where every row has the same value."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 80
        return pd.DataFrame({
            "always_42": np.full(n, 42.0),
            "always_A": ["A"] * n,
            "varying": np.random.randn(n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_constant_columns_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 8. High-cardinality categorical
# ---------------------------------------------------------------------------

class TestHighCardinality:
    """Categorical column where almost every value is unique."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "user_id": [f"user_{i:04d}" for i in range(n)],
            "amount": np.random.uniform(1, 1000, n),
            "status": np.random.choice(["active", "inactive"], n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_high_cardinality_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 9. Mixed NaN patterns
# ---------------------------------------------------------------------------

class TestMixedNaN:
    """Columns with partial NaN / None values in both numeric and categorical."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 100
        prices = np.random.uniform(10, 500, n).astype(float)
        prices[np.random.choice(n, 15, replace=False)] = np.nan

        categories = np.array(np.random.choice(["A", "B", "C"], n), dtype=object)
        categories[np.random.choice(n, 10, replace=False)] = None

        return pd.DataFrame({
            "price": prices,
            "category": categories,
            "complete": np.random.randn(n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_mixed_nan_fit_sample(self, model, df):
        if model in NO_NAN_SUPPORT:
            pytest.skip(f"{model} does not handle NaN values natively")
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 10. Empty-string categorical values
# ---------------------------------------------------------------------------

class TestEmptyStrings:
    """Categorical column that includes empty strings."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 80
        labels = np.random.choice(["alpha", "beta", "", "gamma"], n)
        return pd.DataFrame({
            "label": labels,
            "value": np.random.randn(n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_empty_string_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 11. Boolean-typed columns
# ---------------------------------------------------------------------------

class TestBooleanDtype:
    """DataFrame with native bool dtype columns (not 0/1 int)."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "is_active": np.random.choice([True, False], n),
            "is_verified": np.random.choice([True, False], n),
            "score": np.random.uniform(0, 100, n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_boolean_dtype_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 12. Wide DataFrame (many columns)
# ---------------------------------------------------------------------------

class TestWideDataFrame:
    """DataFrame with many columns (>50) to stress encoding/decoding."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 100
        data = {}
        for i in range(30):
            data[f"num_{i}"] = np.random.randn(n)
        for i in range(20):
            data[f"cat_{i}"] = np.random.choice(["x", "y", "z"], n)
        return pd.DataFrame(data)

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_wide_df_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 13. Mixed dtypes stress test
# ---------------------------------------------------------------------------

class TestMixedDtypes:
    """DataFrame mixing int, float, bool, object, and category dtypes."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "int_col": np.random.randint(0, 100, n),
            "float_col": np.random.uniform(0, 1, n),
            "bool_col": np.random.choice([True, False], n),
            "str_col": np.random.choice(["a", "b", "c"], n),
            "cat_col": pd.Categorical(np.random.choice(["low", "mid", "high"], n)),
        })
        return df

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_mixed_dtypes_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 14. Tensor output validation
# ---------------------------------------------------------------------------

class TestTensorOutput:
    """Ensure tensor output matches expected shape and dtype."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "a": np.random.randn(80),
            "b": np.random.choice(["x", "y"], 80),
            "c": np.random.randint(0, 5, 80),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_tensor_shape_and_dtype(self, model, df):
        config = {}
        synth = TableSynthesizer(model, config)
        synth.fit(df)
        tensor = synth.sample(n=N_SAMPLES)
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor, got {type(tensor)}"
        assert tensor.shape[0] == N_SAMPLES
        # Width should be at least the number of original columns
        assert tensor.shape[1] >= len(df.columns)
        assert tensor.dtype == torch.float32 or tensor.dtype == torch.float64


# ---------------------------------------------------------------------------
# 15. Reproducibility (seeded generation)
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Models with random_state should produce deterministic output."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "a": np.random.randn(80),
            "b": np.random.choice(["x", "y", "z"], 80),
        })

    @pytest.mark.parametrize("model", _parametrize_seeded())
    def test_seeded_determinism(self, model, df):
        """Two runs with same seed should yield identical output."""
        config = {"random_state": 123}
        r1 = _fit_sample(model, df, config)
        r2 = _fit_sample(model, df, config)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# 16. Tiny dataset (minimal rows)
# ---------------------------------------------------------------------------

class TestTinyDataset:
    """DataFrame with very few rows (stress minimum viable training size)."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": ["a", "b", "a", "b", "a"],
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_tiny_dataset_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 17. Large numeric range (extreme values)
# ---------------------------------------------------------------------------

class TestExtremeValues:
    """Columns with very large and very small numeric values."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 80
        return pd.DataFrame({
            "big": np.random.uniform(1e6, 1e9, n),
            "small": np.random.uniform(1e-9, 1e-6, n),
            "mixed": np.concatenate([
                np.random.uniform(-1e6, -1e3, n // 2),
                np.random.uniform(1e3, 1e6, n // 2),
            ]),
            "label": np.random.choice(["pos", "neg"], n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_extreme_values_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))


# ---------------------------------------------------------------------------
# 18. Imbalanced classes
# ---------------------------------------------------------------------------

class TestImbalancedClasses:
    """Categorical column with severe class imbalance (95/5 split)."""

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        n = 100
        labels = np.array(["majority"] * 95 + ["minority"] * 5)
        np.random.shuffle(labels)
        return pd.DataFrame({
            "class": labels,
            "feat1": np.random.randn(n),
            "feat2": np.random.uniform(0, 10, n),
        })

    @pytest.mark.parametrize("model", _parametrize_all())
    def test_imbalanced_fit_sample(self, model, df):
        result = _fit_sample(model, df)
        _assert_basic(result, set(df.columns))
