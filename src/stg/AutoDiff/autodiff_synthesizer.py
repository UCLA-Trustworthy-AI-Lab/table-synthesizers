import numpy as np
import pandas as pd
import torch
import time
from typing import Optional

from ..base import BaseSynthesizer

try:
    from . import process_GQ as pce
    from . import autoencoder as ae
    from . import diffusion as diff
    from . import TabDDPMdiff as TabDiff
    AUTODIFF_AVAILABLE = True
except ImportError:
    AUTODIFF_AVAILABLE = False


class AutoDiffSynthesizer(BaseSynthesizer):
    """
    AutoDiff synthesizer for tabular data generation.
    
    This synthesizer combines an autoencoder with a diffusion model to generate
    synthetic tabular data. It first learns a latent representation using an
    autoencoder, then trains a diffusion model on the latent features.
    Only supports DataFrame input (not DataLoader).
    """
    
    def __init__(self, data_info=None, threshold=0.01, n_epochs=2000, lr=2e-4, 
                 hidden_size=250, num_layers=3, batch_size=50, diff_n_epochs=2000,
                 T=100, sigma=20, num_batches_per_epoch=50, **kwargs):
        if not AUTODIFF_AVAILABLE:
            raise ImportError("AutoDiff dependencies are required for AutoDiffSynthesizer")
        
        super().__init__(data_info=data_info, **kwargs)
        
        # Extract and use config parameters if provided
        # Support both 'epochs' and 'n_epochs' for compatibility
        if 'epochs' in kwargs:
            n_epochs = kwargs['epochs']
        if 'diff_n_epochs' in kwargs:
            diff_n_epochs = kwargs['diff_n_epochs']
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
            
        # AutoDiff parameters with config override support
        self.threshold = threshold
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.diff_n_epochs = diff_n_epochs
        self.T = T
        self.sigma = sigma
        self.num_batches_per_epoch = num_batches_per_epoch
        
        # Training artifacts
        self.ds = None  # (decoder, latent_features, mu, logvar)
        self.score = None
        self.stored_data = None

    def fit(self, data):
        """Sklearn-style fit method."""
        self.train(data)

    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("AutoDiffSynthesizer only supports DataFrame input, not DataLoader")
        
        # Skip base class conversion and handle DataFrame directly
        self.start_threading()

        
    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("AutoDiffSynthesizer only supports DataFrame input, not DataLoader")
        
        # Skip base class conversion and handle DataFrame directly
        self.start_threading()
        # Set seed across libraries if provided
        self.set_seed(self._seed)
        
        self.stored_data = train_data.copy()
        
        print(f"AutoDiff: training on {len(self.stored_data)} samples")
        
        # Set device using base class with fallback for CUDA issues
        try:
            self.set_device()
            device = self.device
        except Exception as e:
            if "CUDA" in str(e):
                print(f"Warning: CUDA not available ({e}), falling back to CPU")
                import torch
                self.device = torch.device('cpu')
                device = self.device
            else:
                raise
        
        # AutoDiff parameters
        eps = 1e-5
        weight_decay = 1e-6
        maximum_learning_rate = 1e-2
        
        # Preprocess and parse the data
        parser = pce.DataFrameParser().fit(train_data, self.threshold)
        
        # Train the autoencoder and get latent features
        self.ds = ae.train_autoencoder(
            train_data, 
            self.hidden_size, 
            self.num_layers, 
            self.lr, 
            weight_decay, 
            self.n_epochs, 
            self.batch_size, 
            self.threshold,
            device=self._device
        )
        latent_features = self.ds[1].detach()
        
        print("Latent feature shape:", latent_features.shape)
        
        # Train the diffusion model on the latent features
        self.score = TabDiff.train_diffusion(
            latent_features, 
            self.T, 
            eps, 
            self.sigma, 
            self.lr,
            self.num_batches_per_epoch, 
            maximum_learning_rate, 
            weight_decay, 
            self.diff_n_epochs, 
            self.batch_size,
            device=self._device
        )
        
        print("AutoDiff training completed!")
        
        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using AutoDiff."""
        if self.ds is None or self.score is None:
            raise RuntimeError("Model must be trained before generating samples")
        
        device = self.device
        
        latent_features = self.ds[1].detach()
        
        # Generate synthetic samples using Euler-Maruyama sampling
        T_sampling = 300  # Sampling time steps
        P = latent_features.shape[1]
        
        start_time = time.time()
        sample = diff.Euler_Maruyama_sampling(self.score, T_sampling, n_samples, P, device)
        # Move to model device for decoding
        sample = sample.to(self._device)
        end_time = time.time()
        
        # Convert generated samples back to the original table format
        print("Sample shape:", sample.shape)
        gen_output = self.ds[0](sample, self.ds[2], self.ds[3])
        # Ensure CPU tensors for downstream numpy conversion
        if 'nums' in gen_output:
            gen_output['nums'] = gen_output['nums'].detach().cpu()
        if 'cats' in gen_output:
            gen_output['cats'] = [x.detach().cpu() for x in gen_output['cats']]
        if 'bins' in gen_output:
            gen_output['bins'] = gen_output['bins'].detach().cpu()
        syn_df = pce.convert_to_table(self.stored_data, gen_output, self.threshold)
        
        print(f"Sampling duration: {end_time - start_time} seconds")
        
        return syn_df
    
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
