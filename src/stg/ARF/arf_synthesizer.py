import numpy as np
import pandas as pd
import torch
from typing import Optional

from ..base import BaseSynthesizer

try:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    SYNTHCITY_AVAILABLE = True
except ImportError:
    SYNTHCITY_AVAILABLE = False


class ARFSynthesizer(BaseSynthesizer):
    """
    ARF (Adversarial Random Forest) synthesizer for tabular data generation.

    This synthesizer uses synthcity's ARF implementation which combines
    adversarial training with random forest-based generation.
    Only supports DataFrame input (not DataLoader).

    Synthcity plugin parameters (passed via config dict):
        num_trees (int): Number of trees in the forest. Default: 30.
        delta (float): Convergence threshold. Default: 0.
        max_iters (int): Maximum training iterations. Default: 10.
        early_stop (bool): Enable early stopping. Default: True.
        verbose (bool): Print progress. Default: True.
        min_node_size (int): Minimum leaf node size. Default: 5.
        random_state (int): Random seed. Default: 0.
        sampling_patience (int): Max retries for schema-valid sampling. Default: 500.
    """

    # Parameters that synthcity's ARF plugin accepts
    _SYNTHCITY_PARAMS = {
        'num_trees', 'delta', 'max_iters', 'early_stop', 'verbose',
        'min_node_size', 'random_state', 'sampling_patience',
    }

    def __init__(self, data_info=None, **kwargs):
        if not SYNTHCITY_AVAILABLE:
            raise ImportError("synthcity package is required for ARFSynthesizer. "
                            "Install it with: pip install synthcity")

        # Extract synthcity-specific params before passing to base class
        self._synthcity_kwargs = {}
        for key in list(kwargs.keys()):
            if key in self._SYNTHCITY_PARAMS:
                self._synthcity_kwargs[key] = kwargs.pop(key)

        super().__init__(data_info=data_info, **kwargs)
        self.model = None
        self.stored_data = None

    def fit(self, data):
        """Sklearn-style fit method."""
        self.train(data)

    # arfpy uses pd.melt(value_name="value", var_name="variable") internally.
    # Columns with those names conflict and raise ValueError during generation.
    _ARF_RESERVED = {"value", "variable"}

    def _safe_rename(self, df):
        """Rename columns that conflict with arfpy reserved names."""
        renames = {}
        for col in df.columns:
            if col in self._ARF_RESERVED:
                safe = f"__arf_{col}__"
                renames[col] = safe
        if renames:
            df = df.rename(columns=renames)
        # Invert for restore
        return df, {v: k for k, v in renames.items()}

    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        import logging
        logger = logging.getLogger(__name__)

        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("ARFSynthesizer only supports DataFrame input, not DataLoader")

        # Skip base class conversion and handle DataFrame directly
        self.start_threading()

        self.stored_data = train_data.copy()

        # Rename columns that conflict with arfpy internals (melt uses "value"/"variable")
        safe_data, self._arf_restore_map = self._safe_rename(train_data)

        # Build synthcity plugin kwargs
        plugin_kwargs = dict(self._synthcity_kwargs)

        if plugin_kwargs:
            logger.info("ARF: using plugin params: %s", plugin_kwargs)

        # Create synthcity loader and train model
        loader = GenericDataLoader(safe_data)
        self.model = Plugins().get("arf", **plugin_kwargs)
        self.model.fit(loader)

        logger.info("ARF: trained on %d samples", len(self.stored_data))

        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using ARF."""
        if self.model is None:
            raise RuntimeError("Model must be trained before generating samples")

        # Generate samples using synthcity (uses renamed columns internally)
        synthetic_loader = self.model.generate(count=n_samples)
        synthetic_df = synthetic_loader.dataframe()

        # Restore original column names if any were renamed for arfpy compatibility
        restore = getattr(self, '_arf_restore_map', {})
        if restore:
            synthetic_df = synthetic_df.rename(columns=restore)

        return synthetic_df
    
    def sample(self, n=None, return_dataframe=False):
        """Generate synthetic samples."""
        if n is None:
            n = len(self.stored_data) if self.stored_data is not None else 100
        
        synthetic_df = self._generate(n)
        
        if return_dataframe:
            return synthetic_df
        else:
            # Convert to tensor format for compatibility
            # First encode categorical columns if any
            encoded_df = self._encode_for_tensor(synthetic_df)
            return torch.tensor(encoded_df.values, dtype=torch.float32)
    
    def _encode_for_tensor(self, df):
        """Encode DataFrame for tensor conversion.

        Normalises every column to float64 so that df.values returns a
        homogeneous numpy array that torch can consume.  In pandas 2.0+ a
        DataFrame with mixed numeric dtypes (bool + int64 + float64) can
        return an object-dtype array from .values, which torch rejects.

        Rules applied per column:
          - pd.Categorical → cast to object first (avoids code-vs-label mismatch)
          - non-numeric (object/string/bool-object) → label-encode → float64
          - numeric (bool, int*, float*) → cast to float64 directly
        """
        encoded_df = df.copy()

        for col in df.columns:
            series = encoded_df[col]
            # Unbox pandas Categorical before any further processing
            if isinstance(series.dtype, pd.CategoricalDtype):
                encoded_df[col] = series.astype(object)
                series = encoded_df[col]
            if not pd.api.types.is_numeric_dtype(series):
                encoded_df[col] = pd.Categorical(series).codes.astype(np.float64)
            else:
                # Homogenise all numeric dtypes (bool, int32, int64, etc.) to
                # float64 so that df.values never returns an object array.
                encoded_df[col] = series.astype(np.float64)

        return encoded_df
    
    def generate(self, n_samples, condition=None):
        """Generate synthetic samples - called by TableSynthesizer.sample()."""
        # Generate decoded synthetic data (with original types)
        synthetic_decoded_df = self._generate(n_samples)
        
        # Create encoded version for tensor compatibility
        synthetic_encoded_df = self._encode_for_tensor(synthetic_decoded_df)
        
        # Store both versions
        self._last_generated_encoded_df = synthetic_encoded_df
        self._last_generated_df = synthetic_decoded_df
        
        # Convert encoded version to tensor for TableSynthesizer compatibility
        return torch.tensor(synthetic_encoded_df.values, dtype=torch.float32)
    
    def decode_samples(self, tensor_samples):
        """Convert tensor samples back to DataFrame - used for return_dataframe=True."""
        # Return the stored DataFrame if available and matches size
        if hasattr(self, '_last_generated_df') and self._last_generated_df.shape[0] == tensor_samples.shape[0]:
            return self._last_generated_df
        else:
            # Fallback: reconstruct DataFrame from tensor (loses original dtypes)
            if self.stored_data is not None:
                columns = self.stored_data.columns
                return pd.DataFrame(tensor_samples.numpy(), columns=columns)
            else:
                return pd.DataFrame(tensor_samples.numpy())