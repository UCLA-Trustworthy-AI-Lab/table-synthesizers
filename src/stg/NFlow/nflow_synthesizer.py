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


class NFlowSynthesizer(BaseSynthesizer):
    """
    NFlow (Normalizing Flow) synthesizer for tabular data generation.

    This synthesizer uses synthcity's Normalizing Flow implementation which learns
    invertible transformations to model the data distribution.
    Only supports DataFrame input (not DataLoader).

    Synthcity plugin parameters (passed via config dict):
        n_iter (int): NFlow training iterations as interpreted by synthcity.
            The outer stg wrapper maps its public ``epochs`` value to
            synthcity's ``n_iter``. Default: 1000.
        n_layers_hidden (int): Hidden layers. Default: 1.
        n_units_hidden (int): Units per hidden layer. Default: 100.
        batch_size (int): Training batch size. Default: 200.
        num_transform_blocks (int): Transform blocks. Default: 1.
        dropout (float): Dropout rate. Default: 0.1.
        batch_norm (bool): Use batch normalization. Default: False.
        num_bins (int): Spline bins. Default: 8.
        lr (float): Learning rate. Default: 0.001.
        encoder_max_clusters (int): Encoding clusters. Default: 10.
        random_state (int): Random seed. Default: 0.
        sampling_patience (int): Max retries for schema-valid sampling. Default: 500.
        device (str): Device for training. Default: "cpu".
        patience (int): Early stopping patience. Default: 5.
    """

    # Parameters that synthcity's NFlow plugin accepts
    _SYNTHCITY_PARAMS = {
        'n_iter', 'n_layers_hidden', 'n_units_hidden', 'batch_size',
        'num_transform_blocks', 'dropout', 'batch_norm', 'num_bins',
        'tail_bound', 'lr', 'apply_unconditional_transform',
        'base_distribution', 'linear_transform_type', 'base_transform_type',
        'encoder_max_clusters', 'random_state', 'sampling_patience',
        'device', 'patience', 'n_iter_min', 'n_iter_print', 'patience_metric',
    }

    def __init__(self, data_info=None, **kwargs):
        if not SYNTHCITY_AVAILABLE:
            raise ImportError("synthcity package is required for NFlowSynthesizer. "
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

    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        import logging
        logger = logging.getLogger(__name__)

        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("NFlowSynthesizer only supports DataFrame input, not DataLoader")

        # Skip base class conversion and handle DataFrame directly
        self.start_threading()

        self.stored_data = train_data.copy()

        # Build synthcity plugin kwargs
        plugin_kwargs = dict(self._synthcity_kwargs)

        # The outer stg API exposes `epochs`; synthcity NFlow consumes that
        # budget through its `n_iter` parameter.
        if hasattr(self, '_epochs') and self._epochs is not None and 'n_iter' not in plugin_kwargs:
            plugin_kwargs['n_iter'] = self._epochs

        # Auto-detect device if not explicitly set by user (CUDA → CPU fallback)
        # Synthcity NFlow accepts "cuda" or "cpu" only (not "mps")
        if 'device' not in plugin_kwargs:
            from ..gpu_utils import detect_best_device
            detected = detect_best_device()
            plugin_kwargs['device'] = "cuda" if detected.type == "cuda" else "cpu"
            logger.info("NFlow: auto-detected device: %s", plugin_kwargs['device'])

        if plugin_kwargs:
            logger.info("NFlow: using plugin params: %s", plugin_kwargs)

        # Create synthcity loader and train model
        loader = GenericDataLoader(train_data)
        self.model = Plugins().get("nflow", **plugin_kwargs)
        self.model.fit(loader)

        logger.info("NFlow: trained on %d samples", len(self.stored_data))

        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using NFlow."""
        if self.model is None:
            raise RuntimeError("Model must be trained before generating samples")
        
        # Generate samples using synthcity
        synthetic_loader = self.model.generate(count=n_samples)
        synthetic_df = synthetic_loader.dataframe()
        
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
