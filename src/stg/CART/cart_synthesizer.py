import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional

from ..base import BaseSynthesizer
from .cart import cart_synthesizer


class CARTSynthesizer(BaseSynthesizer):
    """
    CART-based synthesizer for tabular data generation.
    
    This synthesizer uses Classification and Regression Trees (CART) to learn
    conditional dependencies between columns and generate synthetic data.
    Only supports DataFrame input (not DataLoader).
    """
    
    def __init__(self, data_info=None, max_depth=None, random_state=None, **kwargs):
        super().__init__(data_info=data_info, **kwargs)
        self.max_depth = max_depth
        self.random_state = random_state
        self.trained_data = None
        self.numeric_cols = []
        self.categorical_cols = []
        
    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("CARTSynthesizer only supports DataFrame input, not DataLoader")
        
        # Skip base class conversion and handle DataFrame directly
        self.start_threading()
        
        # Store original data and create encoded version for training
        self.original_data = train_data.copy()
        self.trained_data = self._encode_categorical_data(train_data)
        
        # Determine numeric and categorical columns from original data
        for col in self.original_data.columns:
            if pd.api.types.is_numeric_dtype(self.original_data[col]):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)
        
        print(f"CART: stored {len(self.trained_data)} training samples")
        print(f"Numeric columns: {self.numeric_cols}")
        print(f"Categorical columns: {self.categorical_cols}")
        
        self.stop_threading()
        
    def _encode_categorical_data(self, df):
        """Encode categorical columns to integers for CART training."""
        encoded_df = df.copy()
        self.categorical_encoders = {}
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Encode categorical column
                categories = pd.Categorical(df[col])
                encoded_df[col] = categories.codes
                # Store mapping for decoding
                self.categorical_encoders[col] = list(categories.categories)
        
        return encoded_df
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using CART (decoded version)."""
        synthetic_encoded_df = self._generate_encoded(n_samples)
        return self._decode_categorical_data(synthetic_encoded_df)
    
    def _decode_categorical_data(self, df):
        """Decode categorical columns from integers back to original categories."""
        decoded_df = df.copy()
        
        for col in self.categorical_cols:
            if col in self.categorical_encoders:
                categories = self.categorical_encoders[col]
                # Convert codes back to categories
                codes = df[col].astype(int).clip(0, len(categories) - 1)
                decoded_df[col] = [categories[code] for code in codes]
        
        return decoded_df
    
    def sample(self, n=None, return_dataframe=False):
        """Generate synthetic samples."""
        if n is None:
            n = len(self.trained_data) if self.trained_data is not None else 100
        
        synthetic_df = self._generate(n)
        
        if return_dataframe:
            return synthetic_df
        else:
            # Convert to tensor format for compatibility
            return torch.tensor(synthetic_df.values, dtype=torch.float32)
    
    def generate(self, n_samples, condition=None):
        """Generate synthetic samples - called by TableSynthesizer.sample()."""
        # Generate encoded synthetic data first
        synthetic_encoded_df = self._generate_encoded(n_samples)
        # Generate decoded version for DataFrame output
        synthetic_decoded_df = self._decode_categorical_data(synthetic_encoded_df)
        
        # Store both versions
        self._last_generated_encoded_df = synthetic_encoded_df
        self._last_generated_df = synthetic_decoded_df
        
        # Convert encoded version to tensor for TableSynthesizer compatibility
        return torch.tensor(synthetic_encoded_df.values, dtype=torch.float32)
    
    def _generate_encoded(self, n_samples):
        """Generate synthetic samples in encoded format (integers only)."""
        if self.trained_data is None:
            raise RuntimeError("Model must be trained before generating samples")
        
        # Use the original CART synthesizer function on encoded data
        synthetic_df = cart_synthesizer(
            df_encoded=self.trained_data,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            n_rows=n_samples,
            random_state=self.random_state,
            max_depth=self.max_depth
        )
        
        return synthetic_df
    
    def decode_samples(self, tensor_samples):
        """Convert tensor samples back to DataFrame - used for return_dataframe=True."""
        # Return the stored DataFrame if available and matches size
        if hasattr(self, '_last_generated_df') and self._last_generated_df.shape[0] == tensor_samples.shape[0]:
            return self._last_generated_df
        else:
            # Fallback: reconstruct DataFrame from tensor (loses original dtypes)
            if self.trained_data is not None:
                columns = self.trained_data.columns
                return pd.DataFrame(tensor_samples.numpy(), columns=columns)
            else:
                return pd.DataFrame(tensor_samples.numpy())