import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional

from ..base import BaseSynthesizer
from .smote import sample_smote


class SMOTESynthesizer(BaseSynthesizer):
    """
    SMOTE-based synthesizer for tabular data generation.
    
    This synthesizer uses SMOTE (Synthetic Minority Oversampling Technique) to generate
    synthetic data. It doesn't require traditional "fitting" - just stores dataset info.
    Only supports DataFrame input (not DataLoader).
    """
    
    def __init__(self, data_info=None, target_column=None, k_neighbors=5, 
                 categorical_features=None, frac_samples=1.0, random_state=None, **kwargs):
        super().__init__(data_info=data_info, **kwargs)
        self.target_column = target_column
        self.k_neighbors = k_neighbors
        self.categorical_features = categorical_features
        self.frac_samples = frac_samples
        self.random_state = random_state
        self.stored_data = None
        
    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("SMOTESynthesizer only supports DataFrame input, not DataLoader")
        
        # Skip base class conversion and handle DataFrame directly
        self.start_threading()
        
        self.stored_data = train_data.copy()
        
        # Auto-detect target column if not specified
        if self.target_column is None:
            # For SMOTE, prefer categorical columns since it's a classification method
            categorical_cols = [col for col in self.stored_data.columns 
                              if not pd.api.types.is_numeric_dtype(self.stored_data[col])]
            
            if categorical_cols:
                # Use the first categorical column as target
                self.target_column = categorical_cols[0]
                print(f"Using categorical column '{self.target_column}' as target for SMOTE")
            else:
                # If no categorical columns, bin the last column
                last_col = self.stored_data.columns[-1]
                print(f"No categorical columns found, binning '{last_col}' for SMOTE")
                self.target_column = f'{last_col}_binned'
                # Create binned version with 5 bins
                self.stored_data[self.target_column] = pd.cut(
                    self.stored_data[last_col], bins=5, labels=False
                ).astype(str)
        
        # Auto-detect categorical features if not specified
        if self.categorical_features is None:
            self.categorical_features = [col for col in self.stored_data.columns 
                                       if col != self.target_column and 
                                       not pd.api.types.is_numeric_dtype(self.stored_data[col])]
        
        print(f"SMOTE: stored {len(self.stored_data)} training samples")
        print(f"Target column: {self.target_column}")
        print(f"Categorical features: {self.categorical_features}")
        
        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using SMOTE."""
        if self.stored_data is None:
            raise RuntimeError("Model must be trained before generating samples")
        
        # Handle rare categories that don't meet k_neighbors requirement
        target_counts = self.stored_data[self.target_column].value_counts()
        min_class_size = target_counts.min()
        
        # Adapt k_neighbors for small datasets
        effective_k = min(self.k_neighbors, max(1, min_class_size - 1))
        min_samples_needed = effective_k + 1
        
        # Only filter if there would be enough data remaining
        rare_categories = target_counts[target_counts < min_samples_needed].index
        
        data_for_smote = self.stored_data.copy()
        if len(rare_categories) > 0 and len(rare_categories) < len(target_counts):
            print(f"Removing {len(rare_categories)} rare categories with < {min_samples_needed} samples")
            data_for_smote = data_for_smote[~data_for_smote[self.target_column].isin(rare_categories)]
        elif len(rare_categories) == len(target_counts):
            # All categories are too small, use reduced k_neighbors
            effective_k = max(1, min_class_size - 1)
            print(f"Dataset too small for k_neighbors={self.k_neighbors}, using k_neighbors={effective_k}")
            data_for_smote = self.stored_data.copy()
        
        # Use the sample_smote function with synthetic mode
        synthetic_df = sample_smote(
            df=data_for_smote,
            target=self.target_column,
            categorical_features=self.categorical_features,
            eval_type="synthetic",  # Only return synthetic samples
            k_neighbors=effective_k,
            frac_samples=self.frac_samples,
            seed=self.random_state if self.random_state is not None else 0,
            n_samples=n_samples
        )
        
        return synthetic_df
    
    def _encode_for_tensor(self, df):
        """Encode DataFrame for tensor conversion (similar to CART/DPCART approach)."""
        encoded_df = df.copy()
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Encode categorical column to integers
                categories = pd.Categorical(df[col])
                encoded_df[col] = categories.codes.astype(float)
            else:
                # Ensure numeric columns are float type
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce').astype(float)
        
        # Fill any NaN values with 0
        encoded_df = encoded_df.fillna(0.0)
        
        return encoded_df
    
    def sample(self, n=None, return_dataframe=False):
        """Generate synthetic samples."""
        if n is None:
            n = len(self.stored_data) if self.stored_data is not None else 100
        
        synthetic_df = self._generate(n)
        
        if return_dataframe:
            return synthetic_df
        else:
            # Convert to tensor format for compatibility
            return torch.tensor(synthetic_df.values, dtype=torch.float32)
    
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