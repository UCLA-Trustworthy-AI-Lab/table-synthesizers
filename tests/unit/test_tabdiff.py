import numpy as np
import pandas as pd

from stg.TabDiff.tabdiff_synthesizer import TabDiffSynthesizer


def test_tabdiff_initialization():
    model = TabDiffSynthesizer()
    assert model.stored_data is None


def test_tabdiff_fit_and_sample(sample_data):
    model = TabDiffSynthesizer(random_state=42)
    model.fit(sample_data)

    samples = model.sample(12, return_dataframe=True)
    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (12, sample_data.shape[1])
    assert list(samples.columns) == list(sample_data.columns)


def test_tabdiff_edit(sample_data):
    model = TabDiffSynthesizer(random_state=42)
    model.fit(sample_data)

    row = sample_data.iloc[[0]].copy()
    row["feature1"] = np.nan

    edited = model.edit(row, intervention={"target": "A"}, n_samples=5)
    assert edited.shape == (5, sample_data.shape[1])
    assert (edited["target"] == "A").all()
