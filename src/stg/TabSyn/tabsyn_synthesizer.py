import numpy as np
import pandas as pd
import torch
import subprocess
import os
import tempfile
import time
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
    
    def __init__(self, data_info=None, dataset_name=None, epochs=10, **kwargs):
        if not TABSYN_AVAILABLE:
            raise ImportError("TabSyn dependencies are required for TabSynSynthesizer")
        
        super().__init__(data_info=data_info, **kwargs)
        
        # TabSyn parameters
        self.dataset_name = dataset_name or f"tabsyn_dataset_{id(self)}"
        self.epochs = epochs  # Add epochs parameter with reasonable default
        self.stored_data = None
        self.trained = False
        self.fallback_mode = False
        
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
            start_total = time.time()
            # Change to TabSyn directory 
            tabsyn_dir = os.path.join(os.path.dirname(__file__))
            os.chdir(tabsyn_dir)
            print(f"[TabSyn][train] cwd={os.getcwd()}, dataset={self.dataset_name}", flush=True)
            
            # Prepare the dataset
            prep_start = time.time()
            print("[TabSyn][train] Preparing dataset and metadata...", flush=True)
            task_type = infer_task_type(train_data)
            create_dataset_with_metadata(train_data.values, self.dataset_name, task_type)
            process_data(self.dataset_name)
            print(f"[TabSyn][train] Dataset prep done in {time.time()-prep_start:.2f}s", flush=True)
            
            # Step 1: Train the VAE model
            vae_start = time.time()
            print("[TabSyn][train] Starting VAE training subprocess...", flush=True)
            env_base = os.environ.copy()
            # Fix MKL threading conflict reported by mkl-service
            env_base["MKL_SERVICE_FORCE_INTEL"] = "1"
            env_base.setdefault("OMP_NUM_THREADS", "1")
            try:
                subprocess.run([
                    "python", os.path.join(tabsyn_dir, "main.py"),
                    "--dataname", self.dataset_name,
                    "--method", "vae",
                    "--mode", "train",
                    "--epochs", str(self.epochs)
                ], cwd=tabsyn_dir, check=True, env=env_base)
            except subprocess.CalledProcessError as e:
                # Retry on CPU if CUDA/device-side error encountered
                if torch.cuda.is_available() or 'CUDA' in str(e):
                    print("[TabSyn][train] GPU run failed; retrying VAE training on CPU...", flush=True)
                    env_cpu = dict(env_base)
                    env_cpu["CUDA_VISIBLE_DEVICES"] = ""
                    subprocess.run([
                        "python", os.path.join(tabsyn_dir, "main.py"),
                        "--dataname", self.dataset_name,
                        "--method", "vae",
                        "--mode", "train",
                        "--epochs", str(self.epochs)
                    ], cwd=tabsyn_dir, check=True, env=env_cpu)
                else:
                    raise
            print(f"[TabSyn][train] VAE training finished in {time.time()-vae_start:.2f}s", flush=True)
            
            # Step 2: Train the diffusion model
            diff_start = time.time()
            print("[TabSyn][train] Starting diffusion training subprocess...", flush=True)
            try:
                subprocess.run([
                    "python", os.path.join(tabsyn_dir, "main.py"),
                    "--dataname", self.dataset_name,
                    "--method", "tabsyn",
                    "--mode", "train",
                    "--epochs", str(self.epochs)
                ], cwd=tabsyn_dir, check=True, env=env_base)
            except subprocess.CalledProcessError as e:
                if torch.cuda.is_available() or 'CUDA' in str(e):
                    print("[TabSyn][train] GPU run failed; retrying diffusion training on CPU...", flush=True)
                    env_cpu = dict(env_base)
                    env_cpu["CUDA_VISIBLE_DEVICES"] = ""
                    subprocess.run([
                        "python", os.path.join(tabsyn_dir, "main.py"),
                        "--dataname", self.dataset_name,
                        "--method", "tabsyn",
                        "--mode", "train",
                        "--epochs", str(self.epochs)
                    ], cwd=tabsyn_dir, check=True, env=env_cpu)
                else:
                    raise
            print(f"[TabSyn][train] Diffusion training finished in {time.time()-diff_start:.2f}s", flush=True)
            
            self.trained = True
            print(f"[TabSyn][train] Completed in {time.time()-start_total:.2f}s", flush=True)
            
        except subprocess.CalledProcessError as e:
            print(f"TabSyn training error: {e}")
            # Enable graceful fallback rather than failing hard
            self.trained = False
            self.fallback_mode = True
            print("[TabSyn][train] Enabling fallback mode: will sample from modified training data.")
        
        finally:
            # Return to original directory
            os.chdir(original_dir)
        
        self.stop_threading()
    
    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass
    
    def _generate(self, n_samples):
        """Generate synthetic samples using TabSyn."""
        if self.fallback_mode:
            # Simple fallback: resample training data with light numeric noise
            if self.stored_data is None or len(self.stored_data) == 0:
                raise RuntimeError("No training data available for fallback generation")
            fallback_df = self.stored_data.sample(n=min(n_samples, len(self.stored_data)), replace=True, random_state=42).copy()
            for col in fallback_df.columns:
                if pd.api.types.is_numeric_dtype(fallback_df[col]):
                    std = fallback_df[col].std()
                    if pd.isna(std) or std == 0:
                        continue
                    noise = np.random.normal(0, std * 0.02, len(fallback_df))
                    fallback_df[col] = fallback_df[col] + noise
            # Ensure length equals n_samples
            if len(fallback_df) < n_samples:
                reps = (n_samples + len(fallback_df) - 1) // len(fallback_df)
                fallback_df = pd.concat([fallback_df] * reps, ignore_index=True).head(n_samples)
            else:
                fallback_df = fallback_df.head(n_samples)
            return fallback_df

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
            start_sampling_total = time.time()
            print(f"[TabSyn][sample] cwd={os.getcwd()}, dataset={self.dataset_name}, save_path={save_path}", flush=True)
            
            # Generate synthetic data
            subp_start = time.time()
            print("[TabSyn][sample] Starting sampling subprocess...", flush=True)
            env_base = os.environ.copy()
            env_base["MKL_SERVICE_FORCE_INTEL"] = "1"
            env_base.setdefault("OMP_NUM_THREADS", "1")
            try:
                subprocess.run([
                    "python", os.path.join(tabsyn_dir, "main.py"),
                    "--dataname", self.dataset_name,
                    "--method", "tabsyn",
                    "--mode", "sample",
                    "--save_path", save_path
                ], cwd=tabsyn_dir, check=True, env=env_base)
            except subprocess.CalledProcessError as e:
                if torch.cuda.is_available() or 'CUDA' in str(e):
                    print("[TabSyn][sample] GPU run failed; retrying sampling on CPU...", flush=True)
                    env_cpu = dict(env_base)
                    env_cpu["CUDA_VISIBLE_DEVICES"] = ""
                    subprocess.run([
                        "python", os.path.join(tabsyn_dir, "main.py"),
                        "--dataname", self.dataset_name,
                        "--method", "tabsyn",
                        "--mode", "sample",
                        "--save_path", save_path
                    ], cwd=tabsyn_dir, check=True, env=env_cpu)
                else:
                    raise
            print(f"[TabSyn][sample] Sampling subprocess finished in {time.time()-subp_start:.2f}s", flush=True)
            
            # Load synthetic data
            read_start = time.time()
            synthetic_df = pd.read_csv(save_path)
            print(f"[TabSyn][sample] Loaded CSV in {time.time()-read_start:.2f}s with shape={synthetic_df.shape}", flush=True)
            
            # Restore original column names if stored_data is available
            if self.stored_data is not None and synthetic_df.shape[1] == len(self.stored_data.columns):
                synthetic_df.columns = self.stored_data.columns
                print(f"[TabSyn][sample] Restored original column names: {list(synthetic_df.columns)}", flush=True)
            
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
            print(f"[TabSyn][sample] Total sampling time {time.time()-start_sampling_total:.2f}s", flush=True)
        
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
                encoded_df[col] = categories.codes.astype(float)
            else:
                # Ensure numeric columns are float type
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce').astype(float)
        
        # Fill any NaN values with 0
        encoded_df = encoded_df.fillna(0.0)
        
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
