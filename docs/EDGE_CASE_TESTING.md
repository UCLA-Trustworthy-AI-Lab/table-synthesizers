# Edge Case Testing Guide

This document describes the edge-case test suite in `tests/integration/test_edge_cases.py`. These tests verify that all synthesizer models handle unusual, boundary, and real-world messy data gracefully.

## Model Tiers

Tests run against a representative subset of models from each tier:

| Tier | Models Tested | Rationale |
|------|---------------|-----------|
| **Core** (always available) | Identity, CART, DPCART | No optional dependencies, pure CPU, validates the base `BaseSynthesizer` interface |
| **Torch** (lightweight GPU) | TabDiff, TabPFGen, TVAE | Two new models + one established model; exercises the DataFrame-first training path with PyTorch backends |
| **Optional** (gated by availability) | GaussianCopula, SMOTE | Requires optional packages (sdv / imbalanced-learn); tests `try/except` import gating |

Heavy GPU-only models (CTGAN, TabDDPM, AutoDiff, TabSyn, LTM_VAE) are excluded from edge-case CI to keep the suite fast (<60s on CPU).

## Edge Case Matrix (15 Scenarios)

| # | Test Class | What It Tests | Success Criteria | Known Skips |
|---|-----------|---------------|------------------|-------------|
| 1 | `TestNumericalOnly` | All-float/int DataFrame (3 numeric cols, 100 rows) | Output is numeric, shape matches, column names preserved | None |
| 2 | `TestCategoricalOnly` | All-string DataFrame (3 object cols, 100 rows) | Output shape matches, column names preserved | None |
| 3 | `TestUnicode` | CJK (価格, カテゴリ), German (schön_ü) column names + values | Column names round-trip correctly through encode/decode | None |
| 4 | `TestBinaryData` | bool, 0/1 int, yes/no string columns | All binary representations handled, shape correct | None |
| 5 | `TestNestedJSON` | Serialised JSON string column (`{"key": "v0", "nested": {"a": 5}}`) | JSON treated as opaque categorical, not parsed | None |
| 6 | `TestSingleColumn` | 1-column DataFrame (numeric and categorical variants) | Models handle minimal width | DPCART, TabPFGen (need target + features) |
| 7 | `TestConstantColumns` | Zero-variance columns (always 42.0, always "A") | No division-by-zero crash, shape correct | None |
| 8 | `TestHighCardinality` | ~100 unique categorical values (`user_0000` to `user_0099`) | Handles many categories without OOM or crash | None |
| 9 | `TestMixedNaN` | `np.nan` in numeric + `None` in categorical columns | Models handle or gracefully skip NaN | CART, DPCART, TVAE |
| 10 | `TestEmptyStrings` | `""` as a categorical value alongside normal strings | Empty string is a valid category | None |
| 11 | `TestBooleanDtype` | Native `bool` dtype columns (not 0/1 int) | Bool handled correctly in encode/decode | None |
| 12 | `TestWideDataFrame` | 50 columns (30 numeric + 20 categorical) | Encoding scales, shape preserved, no dimension mismatch | None |
| 13 | `TestMixedDtypes` | int + float + bool + str + `pd.Categorical` in one DataFrame | All dtypes coexist without type errors | None |
| 14 | `TestTensorOutput` | `.sample()` returns `torch.Tensor` (default mode) | Tensor shape `(N, >= n_cols)`, dtype float32 or float64 | None |
| 15 | `TestReproducibility` | Seeded determinism (`random_state=123`) | Two runs with same seed produce identical output | TabPFGen (SGLD stochastic) |

## Universal Success Criteria

Every test (unless skipped) asserts:

```python
assert isinstance(result, pd.DataFrame)         # Correct return type
assert result.shape[0] == N_SAMPLES              # Exactly 10 rows
assert set(result.columns) == set(df.columns)    # Column names preserved
# No unhandled exceptions during fit/sample
```

## Skip Conditions

| Condition | Affected Models | Reason |
|-----------|----------------|--------|
| `MULTI_COLUMN_ONLY` | DPCART, TabPFGen | These models require at least 2 columns (target + features). Single-column tests are skipped. |
| `NO_NAN_SUPPORT` | CART, DPCART, TVAE | These models do not handle NaN/None values natively. NaN tests are skipped. |
| `SEEDED_MODELS` | TabDiff, CART, DPCART only | Reproducibility test only runs on models that accept `random_state`. TabPFGen uses SGLD with inherent stochasticity. |

## Architecture

```
tests/integration/test_edge_cases.py
├── _fit_sample()        # Helper: creates TableSynthesizer, fits, samples DataFrame
├── _assert_basic()      # Helper: checks isinstance, shape, columns
├── _models_for()        # Helper: filters model list by DEFAULT_MODELS availability
├── TestNumericalOnly    # Scenario 1
├── TestCategoricalOnly  # Scenario 2
├── ...
└── TestReproducibility  # Scenario 15
```

Each test class:
1. Defines a `@pytest.fixture` that creates the edge-case DataFrame
2. Uses `@pytest.mark.parametrize("model", _models_for())` to run against all available models
3. Calls `_fit_sample()` and `_assert_basic()` for consistent validation

## Running the Tests

```bash
# Run all edge case tests
pytest tests/integration/test_edge_cases.py -v

# Run only edge case tests (using marker)
pytest -m edge_case -v

# Run a specific scenario
pytest tests/integration/test_edge_cases.py::TestUnicode -v

# Run with specific model filtering (e.g., only core models)
pytest tests/integration/test_edge_cases.py -k "Identity or CART" -v

# Expected: ~86 pass, ~7 skip (skips depend on model availability)
```

## Adding New Edge Cases

1. Add a new numbered section in `test_edge_cases.py`
2. Create a test class following the pattern:
   ```python
   class TestYourScenario:
       """Description of the edge case."""

       @pytest.fixture
       def df(self):
           np.random.seed(42)
           return pd.DataFrame({...})

       @pytest.mark.parametrize("model", _models_for())
       def test_your_scenario(self, model, df):
           result = _fit_sample(model, df)
           _assert_basic(result, set(df.columns))
   ```
3. Update the skip sets (`MULTI_COLUMN_ONLY`, `NO_NAN_SUPPORT`) if your scenario triggers known model limitations
4. Update this document with the new scenario
