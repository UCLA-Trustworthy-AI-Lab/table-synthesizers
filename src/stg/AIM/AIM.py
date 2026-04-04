"""
AIM (Adaptive and Iterative Mechanism) synthesizer for differentially private
tabular data generation.

Wraps SmartNoise's AIMSynthesizer, which implements the full AIM algorithm
from McKenna et al. (2022): "AIM: An Adaptive and Iterative Mechanism for
Differentially Private Synthetic Data" (VLDB 2023).

The algorithm:
  1. Iteratively selects the worst-approximated marginal query
     via the exponential mechanism
  2. Measures it with calibrated Gaussian noise (DP guarantee)
  3. Updates a graphical model (PGM) via mirror descent inference
  4. Repeats until the privacy budget is exhausted
  5. Samples synthetic data from the final graphical model

Dependencies:
  - smartnoise-synth (pip install smartnoise-synth)
  - private-pgm / mbi (pip install git+https://github.com/ryan112358/private-pgm.git@01f02f17)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..base import BaseSynthesizer

try:
    from snsynth import Synthesizer as SNSynthesizer

    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

logger = logging.getLogger(__name__)


# Simplified Dataset class that doesn't require complex mbi internals
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
    def __init__(self, df, domain_dict, weights=None):
        """Simple dataset that works with basic domain info"""
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
        """Return the database in vector-of-counts form"""
        # Create bins for histogramdd
        bins = []
        for col in self.df.columns:
            bins.append(range(self.domain_dict[col] + 1))
        
        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans

class SimpleModel:
    def __init__(self, synthetic_df, domain_dict, data_domain=None):
        self.synthetic_df = synthetic_df
        self.domain_dict = domain_dict
        self.domain = data_domain 
        
    def synthetic_data(self, rows=None):
        if rows is None:
            return SimpleDataset(self.synthetic_df, self.domain_dict)
        else:
            # Sample with replacement if needed
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

class AIM(BaseSynthesizer):
    """
    Differentially private synthesizer using the AIM algorithm.

    Wraps SmartNoise SDK's ``AIMSynthesizer``, which uses the ``mbi``
    (private-pgm) library for graphical model inference.

    Only supports discrete / categorical data. Continuous columns are
    automatically discretized into bins before synthesis.
    """

    def __init__(
        self,
        data_info=None,
        epsilon: float = 1.0,
        delta: float = 1e-9,
        max_model_size: int = 80,
        degree: int = 2,
        num_marginals: Optional[int] = None,
        max_cells: int = 10000,
        rounds: Optional[int] = None,
        preprocessor_eps: float = 0.0,
        n_bins: int = 15,
        prng=None,
        # Legacy compat (accepted, ignored)
        max_iters: int = 1000,
        structural_zeros=None,
        bounded: bool = True,
        checkpoint_interval_seconds: int = 30,
        epochs=None,
        **kwargs,
    ):
        if not AIM_AVAILABLE:
            raise ImportError(
                "AIM requires smartnoise-synth and private-pgm. Install with:\n"
                "  pip install smartnoise-synth\n"
                "  pip install git+https://github.com/ryan112358/private-pgm.git"
                "@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca"
            )

        super().__init__(
            data_info=data_info,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            epochs=epochs or 1,
            **kwargs,
        )

        # AIM parameters
        self.epsilon = epsilon
        self.delta = delta
        self.max_model_size = max_model_size
        self.degree = degree
        self.num_marginals = num_marginals
        self.max_cells = max_cells
        self.rounds = rounds
        self.preprocessor_eps = preprocessor_eps
        self.n_bins = n_bins
        self.prng = prng if prng is not None else np.random

        # State
        self._aim_synth = None  # SmartNoise AIMSynthesizer instance
        self.stored_data: Optional[pd.DataFrame] = None
        self._column_names = []
        self._dtypes: Dict[str, np.dtype] = {}
        self.model = None  # kept for checkpoint compat
        self.model_loaded = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, data, batch_size=32):
        self.train(data, batch_size)

    def sample(self, n, return_dataframe=False):
        """Generate samples and optionally decode them back to a DataFrame."""
        samples = self.generate(n)
        if return_dataframe:
            return self.decode_samples(samples)
        return samples

        self.start_threading()
        st = time.time()
        logger.info("Training AIM model (epsilon=%.2f)...", self.epsilon)

        self.stored_data = train_data.copy()
        self._column_names = list(train_data.columns)
        self._dtypes = {col: train_data[col].dtype for col in train_data.columns}

        # Create SmartNoise AIM synthesizer
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

        # Fit on the DataFrame
        # preprocessor_eps allocates part of the privacy budget for data
        # preprocessing (discretization). 0.0 means no DP preprocessing.
        self._aim_synth.fit(
            train_data,
            preprocessor_eps=self.preprocessor_eps,
        )

        self.model = self._aim_synth  # for checkpoint compat
        self.model_loaded = True

        logger.info("AIM training completed in %.2fs", time.time() - st)
        self.stop_threading()

    def _train(self, train_dataloader):
        """Handle DataLoader input by converting to DataFrame first."""
        all_data = []
        for batch in train_dataloader:
            all_data.append(batch.detach().cpu().numpy())

        train_data = np.concatenate(all_data, axis=0)
        
        # Preserve encoded feature names when they are available so decoding and
        # checkpoint restoration can reconstruct the original schema.
        num_cols = train_data.shape[1]
        if self.feature_names and len(self.feature_names) == num_cols:
            columns = self.feature_names
        else:
            columns = [f'col_{i}' for i in range(num_cols)]
        train_df = pd.DataFrame(train_data, columns=columns)

        print("Data preparation completed")

        # Convert data to AIM format
        data, workload = self.prepare_parameters(train_df)

        # Run the implemented baseline. This is intentionally named separately
        # from the published AIM algorithm to avoid overstating its behavior.
        self.model, self.synthetic_data = self.run_laplace_baseline(data, workload, train_df.shape[0])

        ed = time.time()
        print("AIM training completed in:", ed - st, "seconds")

    def run_laplace_baseline(self, data, workload, n_samples):
        """Run the implemented Laplace-noise baseline.

        Unlike the real AIM algorithm, this method does not measure selected
        marginals and fit a model to them. It perturbs discretized records
        directly and resamples from the resulting noisy table.
        """
        print("Running Laplace-noise AIM baseline...")
        
        # `workload` and `n_samples` are unused in this baseline implementation,
        # but kept in the signature for compatibility with the higher-level API.
        _ = workload
        _ = n_samples
        
        # Add Laplace noise for differential privacy
        noise_scale = 1.0 / self.epsilon
        noisy_data = data.df.values + self.prng.laplace(0, noise_scale, data.df.shape)
        
        # Clip to valid ranges
        for i, col in enumerate(data.df.columns):
            max_val = data.domain_dict[col] - 1
            noisy_data[:, i] = np.clip(noisy_data[:, i], 0, max_val)
            noisy_data[:, i] = np.round(noisy_data[:, i])
        
        # Create synthetic data
        synthetic_df = pd.DataFrame(noisy_data.astype(np.int64), columns=data.df.columns)
        

        
        model = SimpleModel(synthetic_df, data.domain_dict, data.domain)
        synth = model.synthetic_data()
        
        return model, synth

    def run_simple_aim(self, data, workload, n_samples):
        """Backward-compatible alias for the implemented baseline."""
        return self.run_laplace_baseline(data, workload, n_samples)

    def _generate(self, n, condition=None):
        if self._aim_synth is None:
            raise RuntimeError("Model must be trained before generating samples")

        logger.info("Generating %d samples...", n)
        samples_df = self._aim_synth.sample(n)

        # Ensure column names match
        if self._column_names and len(samples_df.columns) == len(self._column_names):
            samples_df.columns = self._column_names

        # Return as tensor (base class interface)
        data = samples_df.to_numpy(dtype=float, na_value=0)
        return torch.tensor(data, dtype=torch.float32)

    def sample(self, n=None, return_dataframe=False):
        if n is None:
            n = len(self.stored_data) if self.stored_data is not None else 100

        if self._aim_synth is None:
            raise RuntimeError("Model must be trained before sampling")

        samples_df = self._aim_synth.sample(int(n))

        # Ensure column names match
        if self._column_names and len(samples_df.columns) == len(self._column_names):
            samples_df.columns = self._column_names

        if return_dataframe:
            return samples_df

        # Encode categorical columns to numeric for tensor output
        encoded_df = samples_df.copy()
        for col in encoded_df.columns:
            if not pd.api.types.is_numeric_dtype(encoded_df[col]):
                encoded_df[col] = pd.Categorical(encoded_df[col]).codes
        data = encoded_df.to_numpy(dtype=float)
        return torch.tensor(data, dtype=torch.float32)

    def generate(self, n_samples, condition=None):
        return self._generate(int(n_samples), condition)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_state(self):
        return {
            "stored_data": self.stored_data,
            "column_names": self._column_names,
            "dtypes": {k: str(v) for k, v in self._dtypes.items()},
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_model_size": self.max_model_size,
            "degree": self.degree,
            "num_marginals": self.num_marginals,
            "max_cells": self.max_cells,
            "rounds": self.rounds,
            "preprocessor_eps": self.preprocessor_eps,
        }

    def load_state(self, checkpoint):
        state = torch.load(checkpoint, weights_only=False)
        self.model = state['model']
        self.encoders = state.get('encoders', {})
        self.feature_names = state.get('feature_names', [])
        self.column_info = state.get('column_info', {})
        self.data_info = state.get('data_info', self.data_info)
        self._discretization_info = state.get('discretization_info', {})
        self.continuous_binning = state.get('continuous_binning', self.continuous_binning)
        self.continuous_bin_count = state.get('continuous_bin_count', self.continuous_bin_count)
        self.continuous_min_bins = state.get('continuous_min_bins', self.continuous_min_bins)
        self.continuous_max_bins = state.get('continuous_max_bins', self.continuous_max_bins)
        self.synthetic_data = self.model.synthetic_data() if self.model is not None else None
        self.model_loaded = True

    # ------------------------------------------------------------------
    # Legacy compat
    # ------------------------------------------------------------------

    def init_model(self, train_data):
        if self.model_loaded:
            return

    def default_params(self):
        """Return default parameters"""
        params = {}
        params['epsilon'] = 1.0
        params['delta'] = 1e-9
        params['max_model_size'] = 80
        params['degree'] = 2
        params['num_marginals'] = None
        params['max_cells'] = 10000
        return params

    def infer_domain(self, df):
        """Infer discrete domain sizes for the baseline discretization step."""
        domain = {}
        print("Inferring domain for AIM!")
        
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                domain[col] = 2
                continue
            
            # For integer-like columns, use the range
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
        print("Dimension of transformed training data", train_data.shape)
        print("Domain:", domain_dict)
        
        # Ensure data is integer-valued for AIM
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
                    'kind': 'discrete',
                    'offset': min_val,
                }
            else:
                discretized, bin_edges = self._discretize_continuous_column(
                    series,
                    domain_dict[col],
                )
                train_data[col] = discretized
                domain_dict[col] = max(2, int(discretized.max()) + 1 if len(discretized) else 2)
                self._discretization_info[col] = {
                    'kind': 'binned',
                    'bin_edges': bin_edges.tolist(),
                    'strategy': self.continuous_binning,
                }
        
        # Create Dataset
        data = SimpleDataset(train_data, domain_dict)

        # Create workload
        workload = list(itertools.combinations(data.domain.attrs, params['degree']))
        workload = [cl for cl in workload if data.domain.size(cl) <= params['max_cells']]
        if params['num_marginals'] is not None:
            workload = [workload[i] for i in self.prng.choice(len(workload), params['num_marginals'], replace=False)]

        workload = [(cl, 1.0) for cl in workload]
        return data, workload

    def decode_samples(self, samples):
        """Decode baseline samples back to the original DataFrame schema."""
        encoded_df = self._restore_encoded_feature_space(samples)
        if self.encoders and self.feature_names and list(encoded_df.columns) == list(self.feature_names):
            return BaseSynthesizer.decode_samples(self, encoded_df[self.feature_names].to_numpy())
        return encoded_df

    def _restore_encoded_feature_space(self, samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        if self.feature_names and samples.shape[1] == len(self.feature_names):
            columns = self.feature_names
        elif self.model is not None and hasattr(self.model, 'synthetic_df'):
            columns = list(self.model.synthetic_df.columns)
        else:
            columns = [f'col_{i}' for i in range(samples.shape[1])]

        encoded_df = pd.DataFrame(samples, columns=columns)
        restored_df = encoded_df.copy()
        domain_dict = getattr(self.model, 'domain_dict', {}) if self.model is not None else {}

        for col in restored_df.columns:
            info = self._discretization_info.get(col)
            if not info:
                continue

            codes = np.rint(restored_df[col].to_numpy()).astype(int)

            if info['kind'] == 'discrete':
                upper = max(0, int(domain_dict.get(col, codes.max() + 1 if len(codes) else 1)) - 1)
                codes = np.clip(codes, 0, upper)
                restored_df[col] = codes + info.get('offset', 0)
            elif info['kind'] == 'binned':
                edges = np.asarray(info.get('bin_edges', []), dtype=float)
                if edges.size < 2:
                    restored_df[col] = 0.0
                    continue
                midpoints = (edges[:-1] + edges[1:]) / 2.0
                codes = np.clip(codes, 0, len(midpoints) - 1)
                restored_df[col] = midpoints[codes]

        return restored_df

    def _validate_binning_configuration(self):
        valid_strategies = {'uniform', 'quantile'}
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
            "stg.AIM.AIM currently uses a Laplace-noise baseline rather than "
            "the full AIM marginal-selection and model-fitting algorithm.",
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

        if self.continuous_binning == 'quantile':
            codes, edges = pd.qcut(
                series,
                q=effective_bins,
                labels=False,
                retbins=True,
                duplicates='drop',
            )
        else:
            codes, edges = pd.cut(
                series,
                bins=effective_bins,
                labels=False,
                retbins=True,
                duplicates='drop',
            )

        codes = pd.Series(codes, index=series.index).fillna(0).astype(int)
        edges = np.asarray(edges, dtype=float)

        if edges.size < 2:
            fill_value = float(series.dropna().iloc[0]) if not series.dropna().empty else 0.0
            edges = np.asarray([fill_value - 0.5, fill_value + 0.5], dtype=float)
            codes = pd.Series(np.zeros(len(series), dtype=int), index=series.index)

        return codes, edges
