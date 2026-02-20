import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from stg.tableSynthesizer import TableSynthesizer


def create_test_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "numeric_col1": np.random.randn(60),
            "numeric_col2": np.random.uniform(0, 1, 60),
            "categorical_col": np.random.choice(["A", "B", "C"], 60),
            "binary_col": np.random.choice([0, 1], 60),
        }
    )


def test_TabDiff_initialization():
    synthesizer = TableSynthesizer("TabDiff", {})
    assert synthesizer.model is not None


def test_TabDiff_dataframe_support():
    df = create_test_dataframe()
    synthesizer = TableSynthesizer("TabDiff", {"random_state": 42})
    synthesizer.fit(df)

    n_samples = 20
    sampled_tensor = synthesizer.sample(n=n_samples)
    assert sampled_tensor.shape[0] == n_samples
    assert sampled_tensor.shape[1] == df.shape[1]

    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    assert isinstance(sampled_df, pd.DataFrame)
    assert sampled_df.shape[0] == n_samples
    assert set(sampled_df.columns) == set(df.columns)


def test_TabDiff_edit_path():
    df = create_test_dataframe()
    synthesizer = TableSynthesizer("TabDiff", {"random_state": 42})
    synthesizer.fit(df)

    row = df.iloc[[0]].copy()
    row["numeric_col2"] = np.nan
    edited = synthesizer.model.edit(row, intervention={"categorical_col": "B"}, n_samples=5)

    assert edited.shape == (5, df.shape[1])
    assert (edited["categorical_col"] == "B").all()
