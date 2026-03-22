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

    def train(self, train_data, batch_size=32):
        if not isinstance(train_data, pd.DataFrame):
            # If DataLoader, delegate to base class which calls _train
            super().train(train_data, batch_size)
            return

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

        data = np.concatenate(all_data, axis=0)
        num_cols = data.shape[1]
        columns = [f"col_{i}" for i in range(num_cols)]
        df = pd.DataFrame(data, columns=columns)

        # Discretize continuous columns for AIM
        for col in df.columns:
            if not (df[col] == df[col].astype(int)).all():
                df[col] = pd.cut(
                    df[col], bins=self.n_bins, labels=False, duplicates="drop"
                )
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = df[col].astype(int)

        self.stored_data = df.copy()
        self._column_names = list(df.columns)
        self._dtypes = {col: df[col].dtype for col in df.columns}

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
        self._aim_synth.fit(df, preprocessor_eps=self.preprocessor_eps)
        self.model = self._aim_synth
        self.model_loaded = True

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
        from pathlib import Path

        state = (
            torch.load(checkpoint, weights_only=False)
            if isinstance(checkpoint, (str, Path))
            else checkpoint
        )
        self.stored_data = state.get("stored_data")
        self._column_names = state.get("column_names", [])
        self._dtypes = {
            k: np.dtype(v) for k, v in state.get("dtypes", {}).items()
        }
        self.epsilon = state.get("epsilon", self.epsilon)
        self.delta = state.get("delta", self.delta)
        self.max_model_size = state.get("max_model_size", self.max_model_size)
        self.degree = state.get("degree", self.degree)

        # Re-train from stored data if available
        if self.stored_data is not None:
            self.train(self.stored_data)
        self.model_loaded = True

    # ------------------------------------------------------------------
    # Legacy compat
    # ------------------------------------------------------------------

    def init_model(self, train_data):
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
