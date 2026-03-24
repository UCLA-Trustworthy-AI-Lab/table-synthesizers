from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from ..base import BaseSynthesizer

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    TABPFN_EXTENSIONS_AVAILABLE = True
except ImportError:
    TABPFN_EXTENSIONS_AVAILABLE = False


class TabPFNUnsupervisedSynthesizer(BaseSynthesizer):
    """
    Unsupervised synthetic data generator using TabPFN-extensions.

    Wraps ``tabpfn_extensions.unsupervised.TabPFNUnsupervisedModel`` which uses
    TabPFN for density estimation and sample generation without requiring a
    target column.
    """

    def __init__(
        self,
        data_info=None,
        t: float = 1.0,
        n_permutations: int = 3,
        device: str = "auto",
        **kwargs,
    ):
        if not TABPFN_EXTENSIONS_AVAILABLE:
            raise ImportError(
                "TabPFNUnsupervisedSynthesizer requires tabpfn and tabpfn-extensions. "
                "Install with: pip install tabpfn tabpfn-extensions"
            )

        super().__init__(data_info=data_info, **kwargs)
        self.t = t
        self.n_permutations = n_permutations
        self._device_str = device

        self.model: Optional[TabPFNUnsupervisedModel] = None
        self.stored_data: Optional[pd.DataFrame] = None
        self._column_names: List[str] = []
        self._dtypes: Dict[str, np.dtype] = {}
        self._categorical_columns: List[str] = []
        self._categorical_indices: List[int] = []
        self._label_encoders: Dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, data):
        self.train(data)

    def train(self, train_data, batch_size=32):
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError(
                "TabPFNUnsupervisedSynthesizer only supports DataFrame input, not DataLoader"
            )

        self.start_threading()

        self.stored_data = train_data.copy()
        self._column_names = list(train_data.columns)
        self._dtypes = {col: train_data[col].dtype for col in train_data.columns}

        # Identify categorical columns
        self._categorical_columns = [
            col
            for col in train_data.columns
            if not pd.api.types.is_numeric_dtype(train_data[col])
        ]
        self._categorical_indices = [
            train_data.columns.get_loc(col) for col in self._categorical_columns
        ]

        # Label-encode categoricals for the model
        X = train_data.copy()
        self._label_encoders = {}
        for col in self._categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self._label_encoders[col] = le

        # Fill NaN with column medians
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)

        X_numpy = X.to_numpy(dtype=np.float64)

        # Build the unsupervised model
        device = self._resolve_device()
        clf = TabPFNClassifier(device=device)
        reg = TabPFNRegressor(device=device)
        self.model = TabPFNUnsupervisedModel(clf, reg)

        if self._categorical_indices:
            self.model.set_categorical_features(self._categorical_indices)

        self.model.fit(X_numpy)
        self.stop_threading()

    def _train(self, train_data):
        pass

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self, n_samples: int, condition=None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be trained before generating samples")

        synthetic_array = self.model.generate_synthetic_data(
            n_samples,
            t=self.t,
            n_permutations=self.n_permutations,
        )

        # Handle torch tensor or numpy array output
        if isinstance(synthetic_array, torch.Tensor):
            synthetic_array = synthetic_array.cpu().numpy()

        synthetic_df = pd.DataFrame(synthetic_array, columns=self._column_names)

        # Reverse label encoding for categoricals
        for col in self._categorical_columns:
            le = self._label_encoders[col]
            rounded = np.clip(
                np.round(synthetic_df[col].values).astype(int),
                0,
                len(le.classes_) - 1,
            )
            synthetic_df[col] = le.inverse_transform(rounded)

        # Restore original dtypes for numeric columns
        for col in self._column_names:
            if col not in self._categorical_columns:
                try:
                    synthetic_df[col] = synthetic_df[col].astype(self._dtypes[col])
                except (ValueError, TypeError):
                    pass

        # Apply condition if provided
        if condition and isinstance(condition, dict):
            for col, val in condition.items():
                if col in synthetic_df.columns:
                    synthetic_df[col] = val

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
        return torch.tensor(
            encoded_df.to_numpy(dtype=float, copy=False), dtype=torch.float32
        )

    def generate(self, n_samples, condition=None):
        synthetic_df = self._generate(int(n_samples), condition=condition)
        synthetic_encoded_df = self._encode_for_tensor(synthetic_df)
        self._last_generated_df = synthetic_df
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
            return pd.DataFrame(tensor_samples.numpy(), columns=self._column_names)
        return pd.DataFrame(tensor_samples.numpy())

    # ------------------------------------------------------------------
    # Editing / Finetuning
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            "stored_data": self.stored_data,
            "column_names": self._column_names,
            "dtypes": {k: str(v) for k, v in self._dtypes.items()},
            "categorical_columns": self._categorical_columns,
            "label_encoders": self._label_encoders,
            "t": self.t,
            "n_permutations": self.n_permutations,
        }

    def load_state(self, checkpoint):
        self.stored_data = checkpoint["stored_data"]
        self._column_names = checkpoint["column_names"]
        self._dtypes = {k: np.dtype(v) for k, v in checkpoint["dtypes"].items()}
        self._categorical_columns = checkpoint["categorical_columns"]
        self._label_encoders = checkpoint["label_encoders"]
        self.t = checkpoint["t"]
        self.n_permutations = checkpoint["n_permutations"]
        # Re-train the model from stored data
        if self.stored_data is not None:
            self.train(self.stored_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        if self._device_str != "auto":
            return self._device_str
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
