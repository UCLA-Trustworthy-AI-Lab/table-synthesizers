from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import torch

from ..base import BaseSynthesizer

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer as SDVGaussianCopulaSynthesizer

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


class GaussianCopulaSynthesizer(BaseSynthesizer):
    """
    SDV GaussianCopula wrapper with TableSynthesizer-compatible interfaces.
    """

    _SDV_PARAMS = {
        "enforce_min_max_values",
        "enforce_rounding",
        "locales",
        "default_distribution",
        "numerical_distributions",
    }

    def __init__(self, data_info=None, **kwargs):
        if not SDV_AVAILABLE:
            raise ImportError("sdv package is required for GaussianCopulaSynthesizer")

        self._sdv_kwargs = {}
        for key in list(kwargs.keys()):
            if key in self._SDV_PARAMS:
                self._sdv_kwargs[key] = kwargs.pop(key)

        super().__init__(data_info=data_info, **kwargs)
        self.model: Optional[SDVGaussianCopulaSynthesizer] = None
        self.metadata: Optional[SingleTableMetadata] = None
        self.stored_data: Optional[pd.DataFrame] = None

    def fit(self, data):
        self.train(data)

    def train(self, train_data, batch_size=32):
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("GaussianCopulaSynthesizer only supports DataFrame input, not DataLoader")

        self.start_threading()
        self.stored_data = train_data.copy()

        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data=train_data)

        self.model = SDVGaussianCopulaSynthesizer(self.metadata, **self._sdv_kwargs)
        self.model.fit(train_data)

        self.stop_threading()

    def _train(self, train_data):
        pass

    def _sample_with_condition(self, n_samples: int, condition: Dict[str, object]) -> pd.DataFrame:
        assert self.model is not None

        try:
            from sdv.sampling import Condition

            cond = Condition(num_rows=n_samples, column_values=condition)
            sampled = self.model.sample_from_conditions([cond])
            if sampled is None or len(sampled) == 0:
                raise RuntimeError("No rows returned from conditional sampling")
            if len(sampled) < n_samples:
                extra = self.model.sample(num_rows=n_samples - len(sampled))
                sampled = pd.concat([sampled, extra], ignore_index=True)
            sampled = sampled.head(n_samples)
        except Exception:
            sampled = self.model.sample(num_rows=n_samples)
            for col, val in condition.items():
                if col in sampled.columns:
                    sampled[col] = val

        return sampled

    def _generate(self, n_samples: int, condition: Optional[Dict[str, object]] = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be trained before generating samples")

        if condition:
            synthetic_df = self._sample_with_condition(n_samples, condition)
        else:
            synthetic_df = self.model.sample(num_rows=n_samples)

        if self.stored_data is not None:
            synthetic_df = synthetic_df[self.stored_data.columns]

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

        encoded_df = self._encode_for_tensor(synthetic_df)
        return torch.tensor(encoded_df.to_numpy(dtype=float, copy=False), dtype=torch.float32)

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
