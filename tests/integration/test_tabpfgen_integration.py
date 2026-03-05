import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

pytestmark = pytest.mark.gpu

from stg.tableSynthesizer import TableSynthesizer


def create_test_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "numeric_col1": np.random.randn(50),
            "numeric_col2": np.random.uniform(0, 1, 50),
            "categorical_col": np.random.choice(["A", "B", "C"], 50),
            "target": np.random.choice(["yes", "no"], 50),
        }
    )


def test_TabPFGen_initialization():
    synthesizer = TableSynthesizer("TabPFGen", {"n_sgld_steps": 2})
    assert synthesizer.model is not None


def test_TabPFGen_dataframe_support():
    df = create_test_dataframe()
    synthesizer = TableSynthesizer("TabPFGen", {"n_sgld_steps": 3, "random_state": 42})
    synthesizer.fit(df)

    n_samples = 15
    sampled_tensor = synthesizer.sample(n=n_samples)
    assert sampled_tensor.shape[0] == n_samples
    assert sampled_tensor.shape[1] == df.shape[1]

    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    assert isinstance(sampled_df, pd.DataFrame)
    assert sampled_df.shape[0] == n_samples
    assert set(sampled_df.columns) == set(df.columns)


def test_TabPFGen_edit_path():
    df = create_test_dataframe()
    synthesizer = TableSynthesizer("TabPFGen", {"n_sgld_steps": 2, "random_state": 42})
    synthesizer.fit(df)

    row = df.iloc[[0]].copy()
    row["numeric_col1"] = np.nan
    edited = synthesizer.model.edit(row, intervention={"target": "yes"}, n_samples=4)

    assert edited.shape == (4, df.shape[1])
    assert (edited["target"] == "yes").all()
