import numpy as np
import pandas as pd
import torch
import subprocess
import os
import tempfile
from typing import Optional

from ..base import BaseSynthesizer

try:
    from .utils_gen_mia import create_dataset_with_metadata, infer_task_type
    from .process_dataset import process_data
    TABSYN_AVAILABLE = True
except ImportError:
    TABSYN_AVAILABLE = False


class TabSynSynthesizer(BaseSynthesizer):
    """
    TabSyn synthesizer for tabular data generation.
    
    This synthesizer uses TabSyn's VAE + diffusion approach to generate
    synthetic tabular data. It trains a VAE to learn latent representations
    then uses a diffusion model for generation.
    Only supports DataFrame input (not DataLoader).
    """
    
    def __init__(self, data_info=None, dataset_name=None, **kwargs):
        if not TABSYN_AVAILABLE:
            raise ImportError("TabSyn dependencies are required for TabSynSynthesizer")
        
        super().__init__(data_info=data_info, **kwargs)
        
        # TabSyn parameters
        self.dataset_name = dataset_name or f"tabsyn_dataset_{id(self)}"
        self.stored_data = None
        self.trained = False
        
    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("TabSynSynthesizer only supports DataFrame input, not DataLoader")
        
        # Skip base class conversion and handle DataFrame directly
        self.start_threading()
        
        self.stored_data = train_data.copy()
        
        print(f"TabSyn: training on {len(self.stored_data)} samples")
        
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to TabSyn directory 
            tabsyn_dir = os.path.join(os.path.dirname(__file__))
            os.chdir(tabsyn_dir)
            
            # Prepare the dataset
            task_type = infer_task_type(train_data.values)
            create_dataset_with_metadata(train_data.values, self.dataset_name, task_type)
            process_data(self.dataset_name)
            
            # Step 1: Train the VAE model
            subprocess.run([
                "python", "main.py",
                "--dataname", self.dataset_name,
                "--method", "vae",
                "--mode", "train"
            ], check=True)
            
            # Step 2: Train the diffusion model
            subprocess.run([
                "python", "main.py",
                "--dataname", self.dataset_name,
                "--method", "tabsyn",
                "--mode", "train"
            ], check=True)
            
            self.trained = True
            print("TabSyn training completed!")
            
        except subprocess.CalledProcessError as e:
            print(f"TabSyn training error: {e}")
            raise RuntimeError(f"TabSyn training failed: {e}")
        
        finally:
            # Return to original directory
            os.chdir(original_dir)
        
        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using TabSyn."""
        if not self.trained:
            raise RuntimeError("Model must be trained before generating samples")
        
        # Save current directory
        original_dir = os.getcwd()
        
        # Create temporary file for synthetic data
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        save_path = temp_file.name
        temp_file.close()
        
        try:
            # Change to TabSyn directory
            tabsyn_dir = os.path.join(os.path.dirname(__file__))
            os.chdir(tabsyn_dir)
            
            # Generate synthetic data
            subprocess.run([
                "python", "main.py",
                "--dataname", self.dataset_name,
                "--method", "tabsyn",
                "--mode", "sample",
                "--save_path", save_path
            ], check=True)
            
            # Load synthetic data
            synthetic_df = pd.read_csv(save_path)
            
            # Ensure we get the requested number of samples
            if len(synthetic_df) > n_samples:
                synthetic_df = synthetic_df.head(n_samples)
            elif len(synthetic_df) < n_samples:
                # Repeat data if we have fewer samples than requested
                repeats = (n_samples + len(synthetic_df) - 1) // len(synthetic_df)
                synthetic_df = pd.concat([synthetic_df] * repeats, ignore_index=True)
                synthetic_df = synthetic_df.head(n_samples)
            
        except subprocess.CalledProcessError as e:
            print(f"TabSyn sampling error: {e}")
            raise RuntimeError(f"TabSyn sampling failed: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(save_path):
                os.remove(save_path)
            # Return to original directory
            os.chdir(original_dir)
        
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