from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ..base import BaseSynthesizer

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


class TabPFGenSynthesizer(BaseSynthesizer):
    """
    TabPFGen-style synthesizer adapter.

    This implementation ports the SGLD energy core from the upstream TabPFGen
    repository and adapts it to this project's DataFrame fit/sample/edit
    workflow. TabPFN refinement is optional and used when available.
    """

    def __init__(
        self,
        data_info=None,
        n_sgld_steps: int = 25,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        feature_noise_init: float = 0.01,
        target_column: Optional[str] = None,
        classification_max_unique: int = 20,
        use_tabpfn_refinement: bool = True,
        random_state: Optional[int] = None,
        device: str | torch.device | None = "auto",
        **kwargs,
    ):
        super().__init__(data_info=data_info, **kwargs)

        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.feature_noise_init = feature_noise_init
        self.target_column = target_column
        self.classification_max_unique = classification_max_unique
        self.use_tabpfn_refinement = use_tabpfn_refinement
        self.random_state = random_state if random_state is not None else self._seed
        self.device = self._infer_device(device)
        self.scaler = StandardScaler()

        self.stored_data: Optional[pd.DataFrame] = None
        self.original_dtypes: Optional[pd.Series] = None
        self.feature_columns: list[str] = []
        self.feature_numeric_cols: list[str] = []
        self.feature_categorical_cols: list[str] = []
        self.feature_categories: Dict[str, np.ndarray] = {}
        self.target_categories: Optional[np.ndarray] = None
        self.is_classification_target = True
        self._x_train_scaled: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._trained = False

    # Adapted from upstream TabPFGen (src/tabpfgen/tabpfgen.py, MIT License)
    def _infer_device(self, device: str | torch.device | None) -> torch.device:
        if (device is None) or (isinstance(device, str) and device == "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, torch.device):
            return device
        raise ValueError(f"Invalid device: {device}")

    def fit(self, data):
        self.train(data)

    def train(self, train_data, batch_size=32):
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("TabPFGenSynthesizer only supports DataFrame input, not DataLoader")

        self.start_threading()
        self.set_seed(self._seed)

        self.stored_data = train_data.copy()
        self.original_dtypes = self.stored_data.dtypes.copy()

        if self.target_column is None:
            self.target_column = self.stored_data.columns[-1]
        if self.target_column not in self.stored_data.columns:
            raise ValueError(f"target_column '{self.target_column}' not found in training data")

        self.feature_columns = [c for c in self.stored_data.columns if c != self.target_column]
        self.feature_numeric_cols = [
            col for col in self.feature_columns if pd.api.types.is_numeric_dtype(self.stored_data[col])
        ]
        self.feature_categorical_cols = [
            col for col in self.feature_columns if col not in self.feature_numeric_cols
        ]

        x_df = pd.DataFrame(index=self.stored_data.index)
        for col in self.feature_numeric_cols:
            values = pd.to_numeric(self.stored_data[col], errors="coerce")
            x_df[col] = values.fillna(values.mean())

        self.feature_categories = {}
        for col in self.feature_categorical_cols:
            cat = pd.Categorical(self.stored_data[col])
            categories = np.array(cat.categories.tolist(), dtype=object)
            self.feature_categories[col] = categories
            x_df[col] = cat.codes.astype(float)

        x_values = x_df[self.feature_columns].to_numpy(dtype=float)
        self._x_train_scaled = self.scaler.fit_transform(x_values)

        y_series = self.stored_data[self.target_column]
        numeric_target = pd.api.types.is_numeric_dtype(y_series)
        unique_count = int(y_series.nunique(dropna=True))
        self.is_classification_target = (not numeric_target) or (unique_count <= self.classification_max_unique)

        if self.is_classification_target:
            y_cat = pd.Categorical(y_series)
            self.target_categories = np.array(y_cat.categories.tolist(), dtype=object)
            self._y_train = y_cat.codes.astype(int)
        else:
            self.target_categories = None
            y_values = pd.to_numeric(y_series, errors="coerce")
            self._y_train = y_values.fillna(y_values.mean()).to_numpy(dtype=float)

        self._trained = True
        self.stop_threading()

    def _train(self, train_data):
        pass

    # Adapted from upstream TabPFGen (src/tabpfgen/tabpfgen.py, MIT License)
    def _compute_energy(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        distances = torch.cdist(x_synth, x_train)
        min_distances, _ = distances.min(dim=1)

        if torch.is_floating_point(y_train):
            return min_distances

        class_mask = y_synth.unsqueeze(1) == y_train.unsqueeze(0)
        class_distances = distances * class_mask.float()
        class_distances = class_distances.sum(dim=1) / (class_mask.float().sum(dim=1) + 1e-6)
        return min_distances + class_distances

    # Adapted from upstream TabPFGen (src/tabpfgen/tabpfgen.py, MIT License)
    def _sgld_step(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        x_synth = x_synth.clone().detach().requires_grad_(True)
        energy = self._compute_energy(x_synth, y_synth, x_train, y_train)
        energy_sum = energy.sum()

        grad = torch.autograd.grad(
            energy_sum,
            x_synth,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(x_synth)

        noise = torch.randn_like(x_synth) * np.sqrt(2 * self.sgld_step_size)
        return x_synth - self.sgld_step_size * grad + self.sgld_noise_scale * noise

    def _nearest_neighbor_target(self, x_synth_scaled: np.ndarray) -> np.ndarray:
        assert self._x_train_scaled is not None
        assert self._y_train is not None

        x_train_t = torch.tensor(self._x_train_scaled, dtype=torch.float32, device=self.device)
        x_synth_t = torch.tensor(x_synth_scaled, dtype=torch.float32, device=self.device)
        distances = torch.cdist(x_synth_t, x_train_t)
        nearest_idx = distances.argmin(dim=1).cpu().numpy()
        return self._y_train[nearest_idx]

    def _generate_target(self, x_synth_scaled: np.ndarray) -> np.ndarray:
        assert self._x_train_scaled is not None
        assert self._y_train is not None

        if self.is_classification_target:
            if self.use_tabpfn_refinement and TABPFN_AVAILABLE:
                try:
                    clf = TabPFNClassifier(device=self.device)
                    clf.fit(self._x_train_scaled, self._y_train.astype(int))
                    probs = clf.predict_proba(x_synth_scaled)
                    return np.argmax(probs, axis=1).astype(int)
                except Exception:
                    pass
            return self._nearest_neighbor_target(x_synth_scaled).astype(int)

        if self.use_tabpfn_refinement and TABPFN_AVAILABLE:
            try:
                reg = TabPFNRegressor(device=self.device)
                reg.fit(self._x_train_scaled, self._y_train.astype(float))
                pred = reg.predict(x_synth_scaled)
                return np.asarray(pred, dtype=float)
            except Exception:
                pass
        return self._nearest_neighbor_target(x_synth_scaled).astype(float)

    def _decode_feature_column(self, col: str, values: np.ndarray) -> pd.Series:
        assert self.original_dtypes is not None
        dtype = self.original_dtypes[col]

        if col in self.feature_categorical_cols:
            categories = self.feature_categories[col]
            idx = np.clip(np.rint(values).astype(int), 0, max(len(categories) - 1, 0))
            decoded = pd.Series(categories[idx], name=col)
            return decoded.astype(dtype, copy=False) if not isinstance(dtype, pd.CategoricalDtype) else pd.Series(
                pd.Categorical(decoded, categories=dtype.categories), name=col
            )

        series = pd.Series(values, name=col)
        if pd.api.types.is_bool_dtype(dtype):
            return (series > 0.5).astype(bool)
        if pd.api.types.is_integer_dtype(dtype):
            return np.round(series).astype(dtype)
        if pd.api.types.is_float_dtype(dtype):
            return series.astype(dtype)
        return series.astype(dtype, copy=False)

    def _decode_target_column(self, y_values: np.ndarray) -> pd.Series:
        assert self.target_column is not None
        assert self.original_dtypes is not None

        dtype = self.original_dtypes[self.target_column]
        if self.is_classification_target:
            assert self.target_categories is not None
            idx = np.clip(np.rint(y_values).astype(int), 0, max(len(self.target_categories) - 1, 0))
            decoded = pd.Series(self.target_categories[idx], name=self.target_column)
            if isinstance(dtype, pd.CategoricalDtype):
                return pd.Series(pd.Categorical(decoded, categories=dtype.categories), name=self.target_column)
            return decoded.astype(dtype, copy=False)

        series = pd.Series(y_values, name=self.target_column)
        if pd.api.types.is_integer_dtype(dtype):
            return np.round(series).astype(dtype)
        if pd.api.types.is_float_dtype(dtype):
            return series.astype(dtype)
        return series.astype(dtype, copy=False)

    def _generate(self, n_samples: int, condition: Optional[Dict[str, object]] = None) -> pd.DataFrame:
        if not self._trained or self.stored_data is None:
            raise RuntimeError("Model must be trained before generating samples")
        assert self._x_train_scaled is not None
        assert self._y_train is not None

        seed = self.random_state if self.random_state is not None else self._seed
        rng = np.random.default_rng(seed)

        train_x_t = torch.tensor(self._x_train_scaled, dtype=torch.float32, device=self.device)
        if self.is_classification_target:
            train_y_t = torch.tensor(self._y_train.astype(int), device=self.device)
        else:
            train_y_t = torch.tensor(self._y_train.astype(float), dtype=torch.float32, device=self.device)

        init_idx = rng.choice(len(self._x_train_scaled), size=n_samples, replace=True)
        x_init = self._x_train_scaled[init_idx]
        x_init = x_init + rng.normal(0.0, self.feature_noise_init, size=x_init.shape)
        x_synth = torch.tensor(x_init, dtype=torch.float32, device=self.device)

        if self.is_classification_target:
            y_init = train_y_t[torch.tensor(init_idx, device=self.device)]
        else:
            y_init = torch.zeros(n_samples, dtype=torch.float32, device=self.device)

        for _ in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_init, train_x_t, train_y_t)

        x_synth_scaled = x_synth.detach().cpu().numpy()
        y_synth = self._generate_target(x_synth_scaled)
        x_synth_unscaled = self.scaler.inverse_transform(x_synth_scaled)

        syn_df = pd.DataFrame(index=np.arange(n_samples))
        for idx, col in enumerate(self.feature_columns):
            syn_df[col] = self._decode_feature_column(col, x_synth_unscaled[:, idx])

        syn_df[self.target_column] = self._decode_target_column(y_synth)
        syn_df = syn_df[self.stored_data.columns]

        if condition:
            for col, val in condition.items():
                if col in syn_df.columns:
                    syn_df[col] = val

        return syn_df

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
