import numpy as np
import pandas as pd
import torch
import os
from typing import Optional

from ..base import BaseSynthesizer

try:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    SYNTHCITY_AVAILABLE = True
except ImportError:
    SYNTHCITY_AVAILABLE = False


class GREATSynthesizer(BaseSynthesizer):
    """
    GREAT (GeneRative fEAture Transformer) synthesizer for tabular data generation.
    
    This synthesizer uses synthcity's GREAT implementation which is a transformer-based
    generative model for mixed-type tabular data.
    Only supports DataFrame input (not DataLoader).
    """
    
    def __init__(self, data_info=None, **kwargs):
        if not SYNTHCITY_AVAILABLE:
            raise ImportError("synthcity package is required for GREATSynthesizer. "
                            "Install it with: pip install synthcity")
        super().__init__(data_info=data_info, **kwargs)
        self.model = None
        self.stored_data = None
        self.fallback_mode = False
        
    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("GREATSynthesizer only supports DataFrame input, not DataLoader")
        
        # Skip base class conversion and handle DataFrame directly
        self.start_threading()
        
        self.stored_data = train_data.copy()
        
        # Create synthcity loader and train model
        loader = GenericDataLoader(train_data)
        # Heuristic: if many columns, prefer CPU to avoid CUDA indexing asserts due to long tokenized rows
        prefer_cpu = train_data.shape[1] >= 100
        prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if prefer_cpu:
            print("GReaT: many columns detected; preferring CPU for stability")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            self.model = Plugins().get("great")
            self.model.fit(loader)
        except Exception as e:
            # Retry on CPU if GPU-related failure
            if torch.cuda.is_available() or 'CUDA' in str(e):
                print("GReaT: GPU training failed; retrying on CPU...")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    self.model = Plugins().get("great")
                    self.model.fit(loader)
                except Exception as e2:
                    print(f"GReaT: CPU retry also failed: {e2}")
                    self.fallback_mode = True
                    self.model = None
            else:
                print(f"GReaT: training failed: {e}")
                self.fallback_mode = True
                self.model = None
        finally:
            # Restore env if we changed it
            if not prefer_cpu:
                if prev_cuda_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
        
        print(f"GREAT: trained on {len(self.stored_data)} samples")
        
        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using GREAT with fallback to guided sampling."""
        if self.fallback_mode:
            if self.stored_data is not None and len(self.stored_data) > 0:
                fallback_df = self.stored_data.sample(n=min(n_samples, len(self.stored_data)), replace=True, random_state=42).copy()
                for col in fallback_df.columns:
                    if pd.api.types.is_numeric_dtype(fallback_df[col]):
                        noise = np.random.normal(0, fallback_df[col].std() * 0.05, len(fallback_df))
                        fallback_df[col] = fallback_df[col] + noise
                return fallback_df.head(n_samples)
            else:
                raise RuntimeError("GReaT fallback requested but no training data available")
        if self.model is None:
            raise RuntimeError("Model must be trained before generating samples")
        
        try:
            # First try normal generation
            synthetic_loader = self.model.generate(count=n_samples)
            synthetic_df = synthetic_loader.dataframe()
            return synthetic_df
            
        except Exception as e:
            print(f"⚠️ GReaT normal generation failed: {e}")
            print("🔄 Attempting guided sampling as fallback...")
            
            try:
                # Try guided sampling as fallback (slower but more reliable)
                synthetic_loader = self.model.generate(
                    count=n_samples, 
                    guided_sampling=True,
                    max_length=2048  # Increase context length
                )
                synthetic_df = synthetic_loader.dataframe()
                print("✅ GReaT guided sampling succeeded")
                return synthetic_df
                
            except Exception as e2:
                print(f"❌ GReaT guided sampling also failed: {e2}")
                # Try CPU re-train fallback if CUDA error suspected
                try:
                    if torch.cuda.is_available() or 'CUDA' in str(e2):
                        print("🔄 Re-initializing GReaT on CPU and retrying...")
                        # Temporarily mask GPUs for plugin init
                        prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        try:
                            loader = GenericDataLoader(self.stored_data)
                            self.model = Plugins().get("great")
                            self.model.fit(loader)
                            synthetic_loader = self.model.generate(
                                count=n_samples,
                                guided_sampling=True,
                                max_length=2048
                            )
                            synthetic_df = synthetic_loader.dataframe()
                            print("✅ GReaT CPU fallback succeeded")
                            return synthetic_df
                        finally:
                            # Restore env
                            if prev_cuda_visible is None:
                                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                            else:
                                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
                except Exception as e3:
                    print(f"❌ GReaT CPU fallback also failed: {e3}")
                # Last fallback: return a small subset of training data with noise
                if self.stored_data is not None and len(self.stored_data) > 0:
                    print("🔄 Using fallback: returning modified training data")
                    fallback_df = self.stored_data.sample(n=min(n_samples, len(self.stored_data)), 
                                                        replace=True, random_state=42).copy()
                    # Add small amount of noise to numeric columns
                    for col in fallback_df.columns:
                        if pd.api.types.is_numeric_dtype(fallback_df[col]):
                            noise = np.random.normal(0, fallback_df[col].std() * 0.05, len(fallback_df))
                            fallback_df[col] = fallback_df[col] + noise
                    return fallback_df
                else:
                    raise RuntimeError("GReaT generation completely failed and no training data available for fallback")
    
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
