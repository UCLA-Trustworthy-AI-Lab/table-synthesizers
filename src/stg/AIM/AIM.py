"""AIM synthesizer wrapper with optional real snsynth backend.

This module prefers a real AIM implementation via ``snsynth`` when that package
is installed. When it is not available, the exported ``AIM`` class falls back to
the existing lightweight Laplace-noise baseline so the package remains usable.
"""

import itertools
import logging
import time
import warnings

import numpy as np
import pandas as pd
import torch

from ..base import BaseSynthesizer

try:
    from snsynth import Synthesizer as SNSynthesizer

    SNSYNTH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SNSynthesizer = None
    SNSYNTH_AVAILABLE = False

AIM_MBI_AVAILABLE = SNSYNTH_AVAILABLE
AIM_AVAILABLE = True
_AIM_BASELINE_WARNING_EMITTED = False

logger = logging.getLogger(__name__)


class Mechanism:
    """Minimal DP mechanism base used by the compatibility wrapper."""

    def __init__(self, epsilon, delta, bounded, prng):
        self.epsilon = epsilon
        self.delta = delta
        self.bounded = bounded
        self.prng = prng

    def exponential_mechanism(self, errors, eps, sensitivity):
        return list(errors.keys())[0] if errors else None

    def gaussian_noise(self, sigma, size):
        return self.prng.normal(0, sigma, size)


class SimpleDomain:
    """Pickle-safe lightweight domain object."""

    def __init__(self, domain_dict):
        self.attrs = list(domain_dict.keys())
        self.domain_dict = domain_dict

    def size(self, cols):
        if isinstance(cols, str):
            cols = [cols]
        size = 1
        for col in cols:
            size *= self.domain_dict[col]
        return size

    def __len__(self):
        return len(self.attrs)


class SimpleDataset:
    """Simple dataset wrapper holding a DataFrame and finite domain metadata."""

    def __init__(self, df, domain_dict, weights=None):
        self.df = df
        self.domain_dict = domain_dict
        self.weights = weights
        self.attrs = list(domain_dict.keys())
        self.domain = SimpleDomain(domain_dict)

    def project(self, cols):
        if isinstance(cols, str):
            cols = [cols]
        projected_df = self.df[cols].copy()
        projected_domain = {col: self.domain_dict[col] for col in cols}
        return SimpleDataset(projected_df, projected_domain, self.weights)

    def datavector(self, flatten=True):
        bins = []
        for col in self.df.columns:
            bins.append(range(self.domain_dict[col] + 1))

        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans


class SimpleModel:
    """Baseline model that resamples from a noisy synthetic table."""

    def __init__(self, synthetic_df, domain_dict, data_domain=None):
        self.synthetic_df = synthetic_df
        self.domain_dict = domain_dict
        self.domain = data_domain

    def synthetic_data(self, rows=None):
        if rows is None:
            return SimpleDataset(self.synthetic_df, self.domain_dict)

        if rows <= len(self.synthetic_df):
            sampled = self.synthetic_df.sample(n=rows, replace=False)
        else:
            sampled = self.synthetic_df.sample(n=rows, replace=True)
        return SimpleDataset(sampled, self.domain_dict)

    def project(self, cols):
        return self.synthetic_data().project(cols)

    @property
    def cliques(self):
        return [tuple(self.synthetic_df.columns)]


class AIM(Mechanism, BaseSynthesizer):
    """
    AIM wrapper with `snsynth` preferred when available.

    Backend behavior:
    - `backend="auto"`: use real AIM through `snsynth` if installed, otherwise
      fall back to the compatibility baseline.
    - `backend="snsynth"`: require the real AIM backend.
    - `backend="baseline"`: force the compatibility baseline.

    Even with the snsynth backend enabled, this class trains on the encoded
    feature space produced by `BaseSynthesizer` so it remains compatible with
    the rest of this repository's tensor-based interfaces.
    """

    def __init__(
        self,
        data_info=None,
        epsilon=1.0,
        delta=1e-9,
        max_model_size=80,
        degree=2,
        num_marginals=None,
        max_cells=10000,
        rounds=None,
        preprocessor_eps=0.0,
        n_bins=15,
        prng=None,
        max_iters=1000,
        structural_zeros=None,
        bounded=True,
        checkpoint_interval_seconds=30,
        epochs=None,
        continuous_binning="uniform",
        continuous_bin_count=None,
        continuous_min_bins=10,
        continuous_max_bins=100,
        backend="auto",
        **kwargs,
    ):
        if prng is None:
            prng = np.random

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta < 0:
            raise ValueError("delta must be non-negative")

        Mechanism.__init__(self, epsilon, delta, bounded, prng)
        BaseSynthesizer.__init__(
            self,
            data_info=data_info,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            epochs=epochs or 1,
            **kwargs,
        )

        self.epsilon = epsilon
        self.delta = delta
        self.max_model_size = max_model_size
        self.degree = degree
        self.num_marginals = num_marginals
        self.max_cells = max_cells
        self.rounds = rounds
        self.preprocessor_eps = preprocessor_eps
        self.n_bins = n_bins
        self.max_iters = max_iters
        self.structural_zeros = structural_zeros or {}
        self.continuous_binning = continuous_binning
        self.continuous_bin_count = continuous_bin_count
        self.continuous_min_bins = continuous_min_bins
        self.continuous_max_bins = continuous_max_bins

        self.rho = epsilon**2 / (2 * np.log(1 / delta)) if delta > 0 else epsilon

        self.prng = prng
        self.model = None
        self.synthetic_data = None
        self.stored_data = None
        self._discretization_info = {}
        self._sample_columns = []
        self._aim_synth = None
        self.backend_preference = backend
        self.backend = self._resolve_backend(backend)

        self._validate_binning_configuration()
        if self.backend == "baseline":
            self._warn_baseline_implementation()

    def fit(self, data, batch_size=32):
        self.train(data, batch_size)

    def sample(self, n, return_dataframe=False):
        samples = self.generate(n)
        if return_dataframe:
            return self.decode_samples(samples)
        return samples

    def _resolve_backend(self, backend):
        valid_backends = {"auto", "snsynth", "baseline"}
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {sorted(valid_backends)}, got {backend!r}")

        if backend == "auto":
            return "snsynth" if SNSYNTH_AVAILABLE else "baseline"
        if backend == "snsynth" and not SNSYNTH_AVAILABLE:
            raise ImportError(
                "backend='snsynth' was requested, but snsynth is not installed. "
                "Install snsynth or use backend='baseline'."
            )
        return backend

    def _train(self, train_dataloader):
        if self.backend == "snsynth":
            self._train_with_snsynth(train_dataloader)
        else:
            self._train_with_baseline(train_dataloader)

    def _train_with_snsynth(self, train_dataloader):
        """Train with the real AIM backend from snsynth."""
        st = time.time()
        logger.info("Training AIM with snsynth backend...")

        train_df = self._dataloader_to_dataframe(train_dataloader)
        self.stored_data = train_df.copy()
        self._sample_columns = list(train_df.columns)

        self._aim_synth = SNSynthesizer.create(
            "aim",
            epsilon=self.epsilon,
            delta=self.delta,
            max_model_size=self.max_model_size,
            degree=self.degree,
            num_marginals=self.num_marginals,
            max_cells=self.max_cells,
            rounds=self.rounds,
            verbose=False,
        )

        try:
            self._aim_synth.fit(train_df, preprocessor_eps=self.preprocessor_eps)
        except TypeError:
            self._aim_synth.fit(train_df)

        self.model = self._aim_synth
        self.synthetic_data = None
        self.model_loaded = True

        logger.info("snsynth AIM training completed in %.2fs", time.time() - st)

    def _train_with_baseline(self, train_dataloader):
        """Train the fallback Laplace-noise baseline."""
        st = time.time()
        logger.info("Training AIM compatibility baseline...")

        train_df = self._dataloader_to_dataframe(train_dataloader)
        self.stored_data = train_df.copy()
        self._sample_columns = list(train_df.columns)

        data, workload = self.prepare_parameters(train_df)
        self.model, self.synthetic_data = self.run_laplace_baseline(
            data,
            workload,
            train_df.shape[0],
        )
        self.model_loaded = True

        logger.info("AIM compatibility baseline training completed in %.2fs", time.time() - st)

    def _dataloader_to_dataframe(self, train_dataloader):
        all_data = []
        for batch in train_dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            all_data.append(batch.detach().cpu().numpy())

        train_array = np.concatenate(all_data, axis=0)
        num_cols = train_array.shape[1]
        if self.feature_names and len(self.feature_names) == num_cols:
            columns = self.feature_names
        else:
            columns = [f"col_{i}" for i in range(num_cols)]
        return pd.DataFrame(train_array, columns=columns)

    def run_laplace_baseline(self, data, workload, n_samples):
        """Run the implemented Laplace-noise baseline."""
        _ = workload
        _ = n_samples

        noise_scale = 1.0 / self.epsilon
        noisy_data = data.df.values + self.prng.laplace(0, noise_scale, data.df.shape)

        for i, col in enumerate(data.df.columns):
            max_val = data.domain_dict[col] - 1
            noisy_data[:, i] = np.clip(noisy_data[:, i], 0, max_val)
            noisy_data[:, i] = np.round(noisy_data[:, i])

        synthetic_df = pd.DataFrame(noisy_data.astype(np.int64), columns=data.df.columns)
        model = SimpleModel(synthetic_df, data.domain_dict, data.domain)
        synth = model.synthetic_data()
        return model, synth

    def run_simple_aim(self, data, workload, n_samples):
        """Backward-compatible alias for the fallback baseline."""
        return self.run_laplace_baseline(data, workload, n_samples)

    def _generate(self, n, condition=None):
        _ = condition
        if self.model is None:
            raise RuntimeError("Model must be trained before generating samples")

        if self.backend == "snsynth":
            samples_df = self._aim_synth.sample(int(n))
            if self._sample_columns and len(samples_df.columns) == len(self._sample_columns):
                samples_df.columns = self._sample_columns
            data = samples_df.fillna(0).to_numpy(dtype=float)
            return torch.tensor(data, dtype=torch.float32)

        synth = self.model.synthetic_data(rows=n)
        data = synth.df.to_numpy()
        return torch.tensor(data, dtype=torch.float32)

    def get_state(self):
        return {
            "model": self.model,
            "backend": self.backend,
            "backend_preference": self.backend_preference,
            "encoders": getattr(self, "encoders", {}),
            "feature_names": getattr(self, "feature_names", []),
            "column_info": getattr(self, "column_info", {}),
            "data_info": getattr(self, "data_info", None),
            "discretization_info": getattr(self, "_discretization_info", {}),
            "continuous_binning": self.continuous_binning,
            "continuous_bin_count": self.continuous_bin_count,
            "continuous_min_bins": self.continuous_min_bins,
            "continuous_max_bins": self.continuous_max_bins,
            "sample_columns": self._sample_columns,
            "implementation": "snsynth_aim" if self.backend == "snsynth" else "laplace_noise_baseline",
        }

    def load_state(self, checkpoint):
        state = torch.load(checkpoint, weights_only=False)
        self.model = state["model"]
        self.backend = state.get("backend", self.backend)
        self.backend_preference = state.get("backend_preference", self.backend_preference)
        self.encoders = state.get("encoders", {})
        self.feature_names = state.get("feature_names", [])
        self.column_info = state.get("column_info", {})
        self.data_info = state.get("data_info", self.data_info)
        self._discretization_info = state.get("discretization_info", {})
        self.continuous_binning = state.get("continuous_binning", self.continuous_binning)
        self.continuous_bin_count = state.get("continuous_bin_count", self.continuous_bin_count)
        self.continuous_min_bins = state.get("continuous_min_bins", self.continuous_min_bins)
        self.continuous_max_bins = state.get("continuous_max_bins", self.continuous_max_bins)
        self._sample_columns = state.get("sample_columns", [])

        if self.backend == "snsynth":
            self._aim_synth = self.model
            self.synthetic_data = None
        else:
            self._aim_synth = None
            self.synthetic_data = self.model.synthetic_data() if self.model is not None else None

        self.model_loaded = True

    def init_model(self, train_data):
        _ = train_data
        if self.model_loaded:
            return

    def default_params(self):
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_model_size": self.max_model_size,
            "degree": self.degree,
            "num_marginals": self.num_marginals,
            "max_cells": self.max_cells,
        }

    def infer_domain(self, df):
        """Infer discrete domain sizes for the baseline discretization step."""
        domain = {}

        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                domain[col] = 2
                continue

            if self._is_integer_valued(series):
                min_val = int(np.floor(series.min()))
                max_val = int(np.ceil(series.max()))
                domain[col] = max(2, max_val - min_val + 1)
            else:
                unique_count = int(series.nunique(dropna=True))
                domain[col] = self._resolve_continuous_bin_count(unique_count)

        return domain

    def prepare_parameters(self, train_data):
        params = self.default_params()
        domain_dict = self.infer_domain(train_data)

        train_data = train_data.copy()
        self._discretization_info = {}
        for col in train_data.columns:
            series = train_data[col]
            if self._is_integer_valued(series):
                non_null = series.dropna()
                min_val = int(np.floor(non_null.min())) if not non_null.empty else 0
                shifted = series.fillna(min_val).astype(float).round().astype(int) - min_val
                train_data[col] = shifted
                domain_dict[col] = max(2, int(shifted.max()) + 1 if len(shifted) else 2)
                self._discretization_info[col] = {
                    "kind": "discrete",
                    "offset": min_val,
                }
            else:
                discretized, bin_edges = self._discretize_continuous_column(
                    series,
                    domain_dict[col],
                )
                train_data[col] = discretized
                domain_dict[col] = max(2, int(discretized.max()) + 1 if len(discretized) else 2)
                self._discretization_info[col] = {
                    "kind": "binned",
                    "bin_edges": bin_edges.tolist(),
                    "strategy": self.continuous_binning,
                }

        data = SimpleDataset(train_data, domain_dict)
        workload = list(itertools.combinations(data.domain.attrs, params["degree"]))
        workload = [cl for cl in workload if data.domain.size(cl) <= params["max_cells"]]
        if params["num_marginals"] is not None:
            workload = [
                workload[i]
                for i in self.prng.choice(len(workload), params["num_marginals"], replace=False)
            ]

        workload = [(cl, 1.0) for cl in workload]
        return data, workload

    def decode_samples(self, samples):
        """Decode generated samples back to the original DataFrame schema."""
        encoded_df = self._restore_encoded_feature_space(samples)
        if self.encoders and self.feature_names and list(encoded_df.columns) == list(self.feature_names):
            return BaseSynthesizer.decode_samples(self, encoded_df[self.feature_names].to_numpy())
        return encoded_df

    def _restore_encoded_feature_space(self, samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        if self.feature_names and samples.shape[1] == len(self.feature_names):
            columns = self.feature_names
        elif self._sample_columns and samples.shape[1] == len(self._sample_columns):
            columns = self._sample_columns
        elif self.model is not None and hasattr(self.model, "synthetic_df"):
            columns = list(self.model.synthetic_df.columns)
        else:
            columns = [f"col_{i}" for i in range(samples.shape[1])]

        encoded_df = pd.DataFrame(samples, columns=columns)
        if self.backend == "snsynth":
            return encoded_df

        restored_df = encoded_df.copy()
        domain_dict = getattr(self.model, "domain_dict", {}) if self.model is not None else {}

        for col in restored_df.columns:
            info = self._discretization_info.get(col)
            if not info:
                continue

            codes = np.rint(restored_df[col].to_numpy()).astype(int)

            if info["kind"] == "discrete":
                upper = max(0, int(domain_dict.get(col, codes.max() + 1 if len(codes) else 1)) - 1)
                codes = np.clip(codes, 0, upper)
                restored_df[col] = codes + info.get("offset", 0)
            elif info["kind"] == "binned":
                edges = np.asarray(info.get("bin_edges", []), dtype=float)
                if edges.size < 2:
                    restored_df[col] = 0.0
                    continue
                midpoints = (edges[:-1] + edges[1:]) / 2.0
                codes = np.clip(codes, 0, len(midpoints) - 1)
                restored_df[col] = midpoints[codes]

        return restored_df

    def _validate_binning_configuration(self):
        valid_strategies = {"uniform", "quantile"}
        if self.continuous_binning not in valid_strategies:
            raise ValueError(
                f"continuous_binning must be one of {sorted(valid_strategies)}, "
                f"got {self.continuous_binning!r}"
            )
        if self.continuous_bin_count is not None and self.continuous_bin_count < 2:
            raise ValueError("continuous_bin_count must be at least 2 when provided")
        if self.continuous_min_bins < 2:
            raise ValueError("continuous_min_bins must be at least 2")
        if self.continuous_max_bins < self.continuous_min_bins:
            raise ValueError("continuous_max_bins must be greater than or equal to continuous_min_bins")

    def _warn_baseline_implementation(self):
        global _AIM_BASELINE_WARNING_EMITTED
        if _AIM_BASELINE_WARNING_EMITTED:
            return

        warnings.warn(
            "stg.AIM.AIM is using the fallback Laplace-noise baseline because "
            "snsynth is not available or backend='baseline' was requested.",
            RuntimeWarning,
            stacklevel=2,
        )
        _AIM_BASELINE_WARNING_EMITTED = True

    def _is_integer_valued(self, series):
        non_null = series.dropna()
        if non_null.empty:
            return True

        values = non_null.to_numpy(dtype=float, copy=False)
        if not np.all(np.isfinite(values)):
            return False
        return np.allclose(values, np.round(values))

    def _resolve_continuous_bin_count(self, unique_count):
        if self.continuous_bin_count is not None:
            return int(self.continuous_bin_count)

        bounded_unique = max(2, int(unique_count))
        return max(
            self.continuous_min_bins,
            min(self.continuous_max_bins, bounded_unique),
        )

    def _discretize_continuous_column(self, series, num_bins):
        num_bins = max(2, int(num_bins))
        unique_count = max(2, int(series.nunique(dropna=True)))
        effective_bins = min(num_bins, unique_count)

        if self.continuous_binning == "quantile":
            codes, edges = pd.qcut(
                series,
                q=effective_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        else:
            codes, edges = pd.cut(
                series,
                bins=effective_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )

        codes = pd.Series(codes, index=series.index).fillna(0).astype(int)
        edges = np.asarray(edges, dtype=float)

        if edges.size < 2:
            fill_value = float(series.dropna().iloc[0]) if not series.dropna().empty else 0.0
            edges = np.asarray([fill_value - 0.5, fill_value + 0.5], dtype=float)
            codes = pd.Series(np.zeros(len(series), dtype=int), index=series.index)

        return codes, edges
