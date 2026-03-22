"""
TabDiff synthesizer — DDPM-based diffusion model for mixed-type tabular data.

Implements denoising diffusion probabilistic modelling (Ho et al., 2020) adapted
for tables following the ideas in the TabDiff paper (Xu et al., 2024):
  - Gaussian diffusion for numerical columns
  - One-hot encoded Gaussian diffusion for categorical columns
  - Shared MLP denoiser with sinusoidal timestep embedding
  - Early stopping on training loss with configurable patience

Source reference: https://github.com/MinkaiXu/TabDiff
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseSynthesizer
from .tabdiff_ref_utils import get_column_name_mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class MLPDenoiser(nn.Module):
    """MLP that predicts noise eps given (x_t, t).

    Architecture: [x_t || t_emb] -> Linear -> ReLU -> ... -> Linear -> x_dim
    """

    def __init__(self, x_dim: int, hidden_dims: tuple = (256, 256, 256),
                 time_emb_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(time_emb_dim)

        layers: List[nn.Module] = []
        in_dim = x_dim + time_emb_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb], dim=-1))


# ---------------------------------------------------------------------------
# Diffusion schedule helpers
# ---------------------------------------------------------------------------

def _linear_beta_schedule(num_steps: int, beta_start: float = 1e-4,
                          beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def _precompute_schedule(betas: torch.Tensor):
    """Return alpha, alpha_bar, and sqrt variants needed for DDPM."""
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "sqrt_recip_alpha": 1.0 / torch.sqrt(alphas),
        "posterior_variance": betas * (1.0 - torch.cat([torch.tensor([0.0]), alpha_bar[:-1]])) / (1.0 - alpha_bar),
    }


# ---------------------------------------------------------------------------
# TabDiffSynthesizer
# ---------------------------------------------------------------------------

class TabDiffSynthesizer(BaseSynthesizer):
    """
    DDPM-based diffusion synthesizer for mixed-type tabular data.

    Numerical columns undergo standard Gaussian diffusion.
    Categorical columns are one-hot encoded and diffused in continuous
    space; during sampling, argmax recovers discrete categories.
    """

    def __init__(
        self,
        data_info=None,
        target_column: Optional[str] = None,
        # Diffusion hyper-parameters
        num_diffusion_steps: int = 1000,
        hidden_dims: tuple = (256, 256, 256),
        time_emb_dim: int = 128,
        # Training hyper-parameters
        epochs: int = 1000,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 20,
        min_epochs: int = 10,
        # Legacy / compat
        noise_scale: float = 0.05,      # kept for API compat; unused
        covariance_regularization: float = 1e-5,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(data_info=data_info, epochs=epochs, **kwargs)
        self.target_column = target_column

        # Diffusion config
        self.num_diffusion_steps = num_diffusion_steps
        self.hidden_dims = hidden_dims
        self.time_emb_dim = time_emb_dim

        # Training config
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._patience = patience
        self._min_epochs = min_epochs

        # Reproducibility
        self.random_state = random_state if random_state is not None else self._seed

        # State populated during training
        self.stored_data: Optional[pd.DataFrame] = None
        self.original_dtypes: Optional[pd.Series] = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []

        # Per-column encoding metadata
        self._cat_categories: Dict[str, list] = {}   # col -> list of categories
        self._num_means: Optional[np.ndarray] = None
        self._num_stds: Optional[np.ndarray] = None

        # Dimension bookkeeping
        self._x_dim: int = 0
        self._num_dim: int = 0
        self._cat_dim: int = 0

        # Model + schedule (set in train)
        self._model: Optional[MLPDenoiser] = None
        self._schedule: Optional[dict] = None

        self._trained = False

        # TabDiff-style metadata mappings
        self.idx_mapping: Dict[int, int] = {}
        self.inverse_idx_mapping: Dict[int, int] = {}
        self.idx_name_mapping: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API wrappers
    # ------------------------------------------------------------------

    def fit(self, data):
        self.train(data)

    def train(self, train_data, batch_size=None):
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("TabDiffSynthesizer only supports DataFrame input, not DataLoader")

        self.start_threading()
        self.set_seed(self.random_state)
        self.set_device()

        self.stored_data = train_data.copy()
        self.original_dtypes = self.stored_data.dtypes.copy()

        # Identify column types
        self.numeric_cols = [
            c for c in self.stored_data.columns
            if pd.api.types.is_numeric_dtype(self.stored_data[c])
        ]
        self.categorical_cols = [
            c for c in self.stored_data.columns if c not in self.numeric_cols
        ]

        # Build TabDiff-style column mappings
        num_col_idx = [self.stored_data.columns.get_loc(c) for c in self.numeric_cols]
        cat_col_idx = [self.stored_data.columns.get_loc(c) for c in self.categorical_cols]
        target_col_idx = []
        if self.target_column and self.target_column in self.stored_data.columns:
            target_col_idx = [self.stored_data.columns.get_loc(self.target_column)]
        self.idx_mapping, self.inverse_idx_mapping, self.idx_name_mapping = (
            get_column_name_mapping(
                self.stored_data, num_col_idx, cat_col_idx, target_col_idx,
            )
        )

        # --- Encode data into a single continuous tensor ---
        x_parts: list[np.ndarray] = []

        # Numerical: z-score normalise
        self._num_dim = len(self.numeric_cols)
        if self.numeric_cols:
            num_vals = self.stored_data[self.numeric_cols].copy()
            for c in self.numeric_cols:
                if num_vals[c].isna().any():
                    num_vals[c] = num_vals[c].fillna(num_vals[c].median())
            num_np = num_vals.to_numpy(dtype=np.float64)
            self._num_means = num_np.mean(axis=0)
            self._num_stds = num_np.std(axis=0)
            self._num_stds[self._num_stds < 1e-8] = 1.0  # avoid division by zero
            num_np = (num_np - self._num_means) / self._num_stds
            x_parts.append(num_np.astype(np.float32))

        # Categorical: one-hot encode
        self._cat_categories = {}
        self._cat_dim = 0
        for col in self.categorical_cols:
            cats = sorted(self.stored_data[col].dropna().unique().tolist())
            self._cat_categories[col] = cats
            mapping = {v: i for i, v in enumerate(cats)}
            codes = self.stored_data[col].map(mapping).fillna(0).astype(int).values
            onehot = np.zeros((len(codes), len(cats)), dtype=np.float32)
            onehot[np.arange(len(codes)), codes] = 1.0
            x_parts.append(onehot)
            self._cat_dim += len(cats)

        x_all = np.concatenate(x_parts, axis=1) if x_parts else np.zeros((len(train_data), 1), dtype=np.float32)
        self._x_dim = x_all.shape[1]

        # Build diffusion schedule
        betas = _linear_beta_schedule(self.num_diffusion_steps)
        self._schedule = _precompute_schedule(betas)
        # Move schedule tensors to device
        for k, v in self._schedule.items():
            self._schedule[k] = v.to(self._device)

        # Build denoiser network
        self._model = MLPDenoiser(
            self._x_dim, self.hidden_dims, self.time_emb_dim,
        ).to(self._device)

        # --- Training loop ---
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay,
        )

        dataset = TensorDataset(torch.tensor(x_all))
        bs = batch_size if batch_size is not None else self._batch_size
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)

        best_loss = float("inf")
        epochs_no_improve = 0

        self._model.train()
        for epoch in range(1, self._epochs + 1):
            self.current_epoch = epoch
            epoch_loss = 0.0
            n_batches = 0
            for (x0_batch,) in loader:
                x0_batch = x0_batch.to(self._device)
                bsz = x0_batch.shape[0]

                # Sample random timesteps
                t = torch.randint(0, self.num_diffusion_steps, (bsz,), device=self._device)

                # Forward diffusion: q(x_t | x_0) = N(sqrt(a_bar)*x_0, (1-a_bar)*I)
                noise = torch.randn_like(x0_batch)
                sqrt_ab = self._schedule["sqrt_alpha_bar"][t].unsqueeze(-1)
                sqrt_omab = self._schedule["sqrt_one_minus_alpha_bar"][t].unsqueeze(-1)
                x_t = sqrt_ab * x0_batch + sqrt_omab * noise

                # Predict noise
                eps_pred = self._model(x_t, t)

                # MSE loss
                loss = nn.functional.mse_loss(eps_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.current_training_loss = avg_loss

            if epoch % max(1, self._epochs // 20) == 0 or epoch == 1:
                logger.info("TabDiff epoch %d/%d  loss=%.6f", epoch, self._epochs, avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                # Save best model weights
                self._best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epoch >= self._min_epochs and epochs_no_improve >= self._patience:
                logger.info(
                    "TabDiff early stopping at epoch %d (patience=%d, best_loss=%.6f)",
                    epoch, self._patience, best_loss,
                )
                break

        # Restore best weights
        if hasattr(self, "_best_state") and self._best_state:
            self._model.load_state_dict(self._best_state)
            del self._best_state

        self._model.eval()
        self._trained = True
        self.stop_threading()

    # Required by BaseSynthesizer but we override train() directly
    def _train(self, train_data):
        pass

    # ------------------------------------------------------------------
    # Sampling (reverse diffusion)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reverse_diffusion(self, n_samples: int) -> np.ndarray:
        """DDPM ancestral sampling: x_T ~ N(0, I) -> x_0."""
        self._model.eval()
        x = torch.randn(n_samples, self._x_dim, device=self._device)

        for t_idx in reversed(range(self.num_diffusion_steps)):
            t = torch.full((n_samples,), t_idx, device=self._device, dtype=torch.long)
            eps_pred = self._model(x, t)

            beta_t = self._schedule["betas"][t_idx]
            sqrt_recip_alpha = self._schedule["sqrt_recip_alpha"][t_idx]
            sqrt_omab = self._schedule["sqrt_one_minus_alpha_bar"][t_idx]

            # Mean of p(x_{t-1} | x_t)
            x = sqrt_recip_alpha * (x - (beta_t / sqrt_omab) * eps_pred)

            # Add noise for t > 0
            if t_idx > 0:
                sigma = torch.sqrt(self._schedule["posterior_variance"][t_idx])
                x = x + sigma * torch.randn_like(x)

        return x.cpu().numpy()

    # ------------------------------------------------------------------
    # Decoding: continuous tensor -> DataFrame
    # ------------------------------------------------------------------

    def _decode_tensor(self, x: np.ndarray) -> pd.DataFrame:
        """Convert denoised continuous tensor back to a DataFrame."""
        df = pd.DataFrame(index=np.arange(x.shape[0]))
        offset = 0

        # Numerical columns: undo z-score
        for i, col in enumerate(self.numeric_cols):
            vals = x[:, offset + i] * self._num_stds[i] + self._num_means[i]
            df[col] = vals
        offset += self._num_dim

        # Categorical columns: argmax over one-hot slices
        for col in self.categorical_cols:
            cats = self._cat_categories[col]
            k = len(cats)
            logits = x[:, offset:offset + k]
            indices = np.argmax(logits, axis=1)
            df[col] = [cats[idx] for idx in indices]
            offset += k

        # Reorder columns to match original
        if self.stored_data is not None:
            col_order = list(self.stored_data.columns)
        elif hasattr(self, '_column_order') and self._column_order is not None:
            col_order = self._column_order
        else:
            col_order = list(df.columns)
        df = df[col_order]

        # Cast dtypes back
        for col in df.columns:
            df[col] = self._cast_column_dtype(col, df[col])

        return df

    # ------------------------------------------------------------------
    # Generation interface
    # ------------------------------------------------------------------

    def _generate(self, n_samples: int, condition: Optional[Dict[str, object]] = None) -> pd.DataFrame:
        if not self._trained or self._model is None:
            raise RuntimeError("Model must be trained before generating samples")

        raw = self._reverse_diffusion(n_samples)
        df = self._decode_tensor(raw)

        if condition:
            df = self._apply_conditions(df, condition)

        return df

    # ------------------------------------------------------------------
    # Helpers shared with old API
    # ------------------------------------------------------------------

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
            return pd.Series(
                pd.Categorical(values, categories=dtype.categories),
                index=values.index, name=values.name,
            )
        return values

    def _apply_conditions(self, df: pd.DataFrame, condition: Optional[Dict[str, object]]) -> pd.DataFrame:
        if not condition:
            return df
        for col, value in condition.items():
            if col in df.columns:
                df[col] = value
        return df

    def _encode_for_tensor(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = df.copy()
        for col in encoded.columns:
            if not pd.api.types.is_numeric_dtype(encoded[col]):
                encoded[col] = pd.Categorical(encoded[col]).codes
        return encoded

    # ------------------------------------------------------------------
    # Public convenience methods (preserve old API surface)
    # ------------------------------------------------------------------

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
        synthetic_encoded = self._encode_for_tensor(synthetic_df)
        self._last_generated_df = synthetic_df
        self._last_generated_encoded_df = synthetic_encoded
        return torch.tensor(
            synthetic_encoded.to_numpy(dtype=float, copy=False), dtype=torch.float32,
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

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def get_state(self):
        state = {
            "model_state": self._model.state_dict() if self._model else None,
            "x_dim": self._x_dim,
            "num_dim": self._num_dim,
            "cat_dim": self._cat_dim,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "cat_categories": self._cat_categories,
            "num_means": self._num_means,
            "num_stds": self._num_stds,
            "original_dtypes": self.original_dtypes,
            "column_order": list(self.stored_data.columns) if self.stored_data is not None else None,
        }
        return state

    def load_state(self, checkpoint):
        state = torch.load(checkpoint, weights_only=False) if isinstance(checkpoint, str) else checkpoint
        self._x_dim = state["x_dim"]
        self._num_dim = state["num_dim"]
        self._cat_dim = state["cat_dim"]
        self.numeric_cols = state["numeric_cols"]
        self.categorical_cols = state["categorical_cols"]
        self._cat_categories = state["cat_categories"]
        self._num_means = state["num_means"]
        self._num_stds = state["num_stds"]
        self.original_dtypes = state["original_dtypes"]
        self._column_order = state.get("column_order")
        self._model = MLPDenoiser(self._x_dim, self.hidden_dims, self.time_emb_dim)
        self._model.load_state_dict(state["model_state"])
        self._model.eval()

        betas = _linear_beta_schedule(self.num_diffusion_steps)
        self._schedule = _precompute_schedule(betas)
        self.set_device()
        self._model.to(self._device)
        for k, v in self._schedule.items():
            self._schedule[k] = v.to(self._device)

        self._trained = True
        self.model_loaded = True
