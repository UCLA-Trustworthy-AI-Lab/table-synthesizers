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


class BayesianNetworkSynthesizer(BaseSynthesizer):
    """
    Bayesian Network-based synthesizer for tabular data generation.
    
    This synthesizer uses synthcity's Bayesian Network implementation to learn
    conditional dependencies between columns and generate synthetic data.
    Only supports DataFrame input (not DataLoader).
    """
    
    def __init__(self, data_info=None, **kwargs):
        if not SYNTHCITY_AVAILABLE:
            raise ImportError("synthcity package is required for BayesianNetworkSynthesizer. "
                            "Install it with: pip install synthcity")
        super().__init__(data_info=data_info, **kwargs)
        self.model = None
        self.stored_data = None

    def fit(self, data):
        """Sklearn-style fit method."""
        self.train(data)

    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("BayesianNetworkSynthesizer only supports DataFrame input, not DataLoader")
        
        # Skip base class conversion and handle DataFrame directly
        self.start_threading()
        
        self.stored_data = train_data.copy()
        
        # Create synthcity loader and train model
        loader = GenericDataLoader(train_data)
        self.model = Plugins().get("bayesian_network")
        self.model.fit(loader)
        
        print(f"BayesianNetwork: trained on {len(self.stored_data)} samples")
        
        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using Bayesian Network."""
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
        """Encode DataFrame for tensor conversion."""
        encoded_df = df.copy()
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Encode categorical column to integers
                categories = pd.Categorical(df[col])
                encoded_df[col] = categories.codes
        
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