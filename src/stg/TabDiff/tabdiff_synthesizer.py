from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..base import BaseSynthesizer
from .tabdiff_ref_utils import get_column_name_mapping


class TabDiffSynthesizer(BaseSynthesizer):
    """
    TabDiff-style synthesizer adapter for mixed-type tabular data.

    This wrapper reuses TabDiff-style column grouping logic (ported from the
    upstream repository) and provides a lightweight mixed-type sampler that
    matches this project's fit/sample/edit interface.
    """

    def __init__(
        self,
        data_info=None,
        target_column: Optional[str] = None,
        noise_scale: float = 0.05,
        covariance_regularization: float = 1e-5,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(data_info=data_info, **kwargs)
        self.target_column = target_column
        self.noise_scale = noise_scale
        self.covariance_regularization = covariance_regularization
        self.random_state = random_state if random_state is not None else self._seed

        self.stored_data: Optional[pd.DataFrame] = None
        self.original_dtypes: Optional[pd.Series] = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self._numeric_mean: Optional[np.ndarray] = None
        self._numeric_cov: Optional[np.ndarray] = None
        self._categorical_probs: Dict[str, np.ndarray] = {}
        self._categorical_values: Dict[str, np.ndarray] = {}
        self._trained = False

        # TabDiff-style metadata mappings
        self.idx_mapping: Dict[int, int] = {}
        self.inverse_idx_mapping: Dict[int, int] = {}
        self.idx_name_mapping: Dict[int, str] = {}

    def fit(self, data):
        self.train(data)

    def train(self, train_data, batch_size=32):
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("TabDiffSynthesizer only supports DataFrame input, not DataLoader")

        self.start_threading()
        self.set_seed(self._seed)

        self.stored_data = train_data.copy()
        self.original_dtypes = self.stored_data.dtypes.copy()

        self.numeric_cols = [
            col for col in self.stored_data.columns
            if pd.api.types.is_numeric_dtype(self.stored_data[col])
        ]
        self.categorical_cols = [
            col for col in self.stored_data.columns if col not in self.numeric_cols
        ]

        num_col_idx = [self.stored_data.columns.get_loc(c) for c in self.numeric_cols]
        cat_col_idx = [self.stored_data.columns.get_loc(c) for c in self.categorical_cols]
        target_col_idx = []
        if self.target_column is not None and self.target_column in self.stored_data.columns:
            target_col_idx = [self.stored_data.columns.get_loc(self.target_column)]

        self.idx_mapping, self.inverse_idx_mapping, self.idx_name_mapping = get_column_name_mapping(
            self.stored_data,
            num_col_idx=num_col_idx,
            cat_col_idx=cat_col_idx,
            target_col_idx=target_col_idx,
        )

        if self.numeric_cols:
            numeric_df = self.stored_data[self.numeric_cols].copy()
            for col in self.numeric_cols:
                if numeric_df[col].isna().any():
                    numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())

            numeric_values = numeric_df.to_numpy(dtype=float)
            self._numeric_mean = numeric_values.mean(axis=0)

            if numeric_values.shape[1] == 1:
                var = float(np.var(numeric_values[:, 0], ddof=1)) if len(numeric_values) > 1 else 1.0
                self._numeric_cov = np.array([[max(var, self.covariance_regularization)]], dtype=float)
            else:
                cov = np.cov(numeric_values, rowvar=False)
                cov = np.atleast_2d(cov)
                self._numeric_cov = cov + np.eye(cov.shape[0]) * self.covariance_regularization

        self._categorical_probs = {}
        self._categorical_values = {}
        for col in self.categorical_cols:
            categories = pd.Categorical(self.stored_data[col]).categories
            values = np.array(categories.tolist(), dtype=object)
            probs = self.stored_data[col].value_counts(normalize=True).reindex(values, fill_value=0.0).to_numpy(dtype=float)
            if probs.sum() <= 0:
                probs = np.ones(len(values), dtype=float) / max(len(values), 1)
            else:
                probs = probs / probs.sum()
            self._categorical_values[col] = values
            self._categorical_probs[col] = probs

        self._trained = True
        self.stop_threading()

    def _train(self, train_data):
        pass

    def _cast_column_dtype(self, col: str, values: pd.Series) -> pd.Series:
        if self.original_dtypes is None:
            return values

        dtype = self.original_dtypes[col]

        if pd.api.types.is_bool_dtype(dtype):
            if pd.api.types.is_numeric_dtype(values):
                return (values > 0.5).astype(bool)
            return values.astype(bool)

        if pd.api.types.is_integer_dtype(dtype):
            rounded = np.round(pd.to_numeric(values, errors="coerce")).fillna(0)
            return rounded.astype(dtype)

        if pd.api.types.is_float_dtype(dtype):
            return pd.to_numeric(values, errors="coerce").astype(dtype)

        if isinstance(dtype, pd.CategoricalDtype):
            categories = dtype.categories
            return pd.Series(
                pd.Categorical(values, categories=categories),
                index=values.index,
                name=values.name,
            )

        return values.astype(dtype, copy=False)

    def _apply_conditions(self, df: pd.DataFrame, condition: Optional[Dict[str, object]]) -> pd.DataFrame:
        if not condition:
            return df
        for col, value in condition.items():
            if col in df.columns:
                df[col] = value
        return df

    def _generate(self, n_samples: int, condition: Optional[Dict[str, object]] = None) -> pd.DataFrame:
        if not self._trained or self.stored_data is None:
            raise RuntimeError("Model must be trained before generating samples")

        seed = self.random_state if self.random_state is not None else self._seed
        rng = np.random.default_rng(seed)

        synthetic_df = pd.DataFrame(index=np.arange(n_samples))

        if self.numeric_cols:
            assert self._numeric_mean is not None
            assert self._numeric_cov is not None

            if len(self.numeric_cols) == 1:
                std = float(np.sqrt(max(self._numeric_cov[0, 0], self.covariance_regularization)))
                sampled_num = rng.normal(loc=self._numeric_mean[0], scale=std, size=(n_samples, 1))
            else:
                sampled_num = rng.multivariate_normal(
                    mean=self._numeric_mean,
                    cov=self._numeric_cov,
                    size=n_samples,
                    method="eigh",
                )

            if self.noise_scale > 0:
                sampled_num = sampled_num + rng.normal(
                    loc=0.0,
                    scale=self.noise_scale,
                    size=sampled_num.shape,
                )

            synthetic_df[self.numeric_cols] = sampled_num

        for col in self.categorical_cols:
            values = self._categorical_values[col]
            probs = self._categorical_probs[col]
            synthetic_df[col] = rng.choice(values, size=n_samples, p=probs)

        synthetic_df = synthetic_df[list(self.stored_data.columns)]
        synthetic_df = self._apply_conditions(synthetic_df, condition)

        for col in synthetic_df.columns:
            synthetic_df[col] = self._cast_column_dtype(col, synthetic_df[col])

        return synthetic_df

    def _encode_for_tensor(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = df.copy()
        for col in encoded_df.columns:
            if not pd.api.types.is_numeric_dtype(encoded_df[col]):
                encoded_df[col] = pd.Categorical(encoded_df[col]).codes
        return encoded_df

    def sample(self, n=None, return_dataframe=False):
        if n is None:
            n = len(self.stored_data) if self.stored_data is not None else 100

        synthetic_df = self._generate(int(n))
        if return_dataframe:
            return synthetic_df

        encoded = self._encode_for_tensor(synthetic_df)
        return torch.tensor(encoded.to_numpy(dtype=float, copy=False), dtype=torch.float32)

    def generate(self, n_samples, condition=None):
        synthetic_df = self._generate(int(n_samples), condition=condition)
        synthetic_encoded_df = self._encode_for_tensor(synthetic_df)
        self._last_generated_df = synthetic_df
        self._last_generated_encoded_df = synthetic_encoded_df
        return torch.tensor(
            synthetic_encoded_df.to_numpy(dtype=float, copy=False),
            dtype=torch.float32,
        )

    def decode_samples(self, tensor_samples):
        if (
            hasattr(self, "_last_generated_df")
            and self._last_generated_df is not None
            and self._last_generated_df.shape[0] == tensor_samples.shape[0]
        ):
            return self._last_generated_df

        if self.stored_data is not None:
            return pd.DataFrame(tensor_samples.numpy(), columns=self.stored_data.columns)
        return pd.DataFrame(tensor_samples.numpy())

    def edit(self, x_row_df, intervention: dict, meta=None, n_samples=1):
        condition: Dict[str, object] = {}

        if isinstance(x_row_df, pd.DataFrame) and not x_row_df.empty:
            row = x_row_df.iloc[0]
            for col in x_row_df.columns:
                value = row[col]
                if pd.notna(value):
                    condition[col] = value

        if intervention:
            condition.update(intervention)

        return self._generate(n_samples=int(n_samples), condition=condition)

    def finetune(self, df_adapt, meta=None):
        self.train(df_adapt)
        return self

    def tta(self, df_adapt, meta=None):
        return self
