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
        
    def fit(self, data):
        """Sklearn-style fit method."""
        self.train(data)

    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("SMOTESynthesizer only supports DataFrame input, not DataLoader")

        # Skip base class conversion and handle DataFrame directly
        self.start_threading()

        self.stored_data = train_data.copy()

        # Auto-detect target column if not specified (use last column)
        if self.target_column is None:
            self.target_column = self.stored_data.columns[-1]

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

        # Detect if target is continuous (regression) or categorical (classification).
        # Rules:
        #   - float dtype                       -> regression (housing prices, etc.)
        #   - integer dtype with HIGH cardinality -> regression (e.g. california_housing
        #       stores prices as int64; needs to be detected as regression even though
        #       dtype is int)
        #   - integer dtype with LOW cardinality -> classification (e.g. Wine Quality
        #       has integer ratings 0-10)
        #   - object/categorical                -> classification
        # This MUST happen BEFORE the rare-category filter — for regression most
        # unique target values appear only once or twice, so the filter would erase
        # 25-100% of the training data and either crash SMOTE or silently bias the
        # synthesizer toward whichever values happened to repeat.
        target_values = self.stored_data[self.target_column]
        n = len(target_values)
        n_unique = target_values.nunique()
        is_regression = (
            pd.api.types.is_float_dtype(target_values)
            or (
                pd.api.types.is_integer_dtype(target_values)
                and n_unique > max(20, 0.05 * n)
            )
        )

        data_for_smote = self.stored_data.copy()
        if not is_regression:
            # Only filter rare categories for classification: SMOTE needs at least
            # k_neighbors+1 samples per class to interpolate.
            min_samples_needed = self.k_neighbors + 1
            target_counts = self.stored_data[self.target_column].value_counts()
            rare_categories = target_counts[target_counts < min_samples_needed].index
            if len(rare_categories) > 0:
                print(f"Removing {len(rare_categories)} rare categories with < {min_samples_needed} samples")
                data_for_smote = data_for_smote[~data_for_smote[self.target_column].isin(rare_categories)]

        # Use the sample_smote function with synthetic mode
        synthetic_df = sample_smote(
            df=data_for_smote,
            target=self.target_column,
            categorical_features=self.categorical_features,
            eval_type="synthetic",  # Only return synthetic samples
            k_neighbors=self.k_neighbors,
            frac_samples=self.frac_samples,
            is_regression=is_regression,
            seed=self.random_state if self.random_state is not None else 0,
            n_samples=n_samples
        )

        # Some SMOTE paths can return fewer rows than requested (e.g., class
        # constraints after filtering). Normalize to exact requested size.
        if len(synthetic_df) > n_samples:
            synthetic_df = synthetic_df.head(n_samples).reset_index(drop=True)
        elif len(synthetic_df) < n_samples:
            if len(synthetic_df) == 0:
                synthetic_df = data_for_smote.sample(
                    n=n_samples,
                    replace=True,
                    random_state=self.random_state if self.random_state is not None else 0,
                ).reset_index(drop=True)
            else:
                extra = synthetic_df.sample(
                    n=n_samples - len(synthetic_df),
                    replace=True,
                    random_state=self.random_state if self.random_state is not None else 0,
                )
                synthetic_df = pd.concat([synthetic_df, extra], ignore_index=True)
        
        return synthetic_df
    
    def _encode_for_tensor(self, df):
        """Encode DataFrame for tensor conversion (similar to CART/DPCART approach)."""
        encoded_df = df.copy()
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Encode categorical column to integers
                categories = pd.Categorical(df[col])
                encoded_df[col] = categories.codes
        
        return encoded_df
    
    def sample(self, n=None, return_dataframe=False):
        """Generate synthetic samples."""
        if n is None:
            n = len(self.stored_data) if self.stored_data is not None else 100

        synthetic_df = self._generate(n)

        if return_dataframe:
            return synthetic_df
        else:
            # Convert to tensor format for compatibility - encode categorical columns first
            encoded_df = self._encode_for_tensor(synthetic_df)
            return torch.tensor(encoded_df.values, dtype=torch.float32)
    
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
