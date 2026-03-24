"""
DataManager - Unified Data Management for Table Synthesizers
=============================================================

Provides centralized data handling for all synthesis algorithms including:
- Preprocessed data storage
- Model checkpoint management
- Sample storage
- Temporary file management

Usage:
    from stg.data_manager import DataManager

    # Initialize for a specific model
    dm = DataManager('TVAE')

    # Save preprocessed data
    dm.save_preprocessed_data(data_dict, 'training_data')

    # Save model checkpoint
    dm.save_checkpoint(checkpoint_dict, 'epoch_100')

    # Save generated samples
    dm.save_samples(synthetic_df, 'synthetic_1000')
"""

import os
import json
import pickle
import shutil
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging


class DataManager:
    """
    Centralized data management for table synthesizers.

    Handles storage and retrieval of:
    - Preprocessed datasets
    - Model checkpoints
    - Generated samples
    - Temporary files
    """

    def __init__(self, model_name: str, base_data_dir: Optional[str] = None):
        """
        Initialize DataManager for a specific model.

        Args:
            model_name: Name of the synthesizer model (e.g., 'TVAE', 'CTGAN')
            base_data_dir: Base directory for data storage (default: './data')
        """
        self.model_name = model_name
        self.base_data_dir = Path(base_data_dir) if base_data_dir else Path('data')

        # Create model-specific directory structure
        self.model_dir = self.base_data_dir / model_name
        self.preprocessed_dir = self.model_dir / 'preprocessed'
        self.checkpoints_dir = self.model_dir / 'checkpoints'
        self.samples_dir = self.model_dir / 'samples'
        self.temp_dir = self.model_dir / 'temp'

        # Create directories
        self._create_directories()

        # Logger
        self.logger = logging.getLogger(__name__)

    def _create_directories(self):
        """Create necessary directory structure."""
        for directory in [self.preprocessed_dir, self.checkpoints_dir,
                         self.samples_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # Preprocessed Data Methods
    # ============================================================================

    def save_preprocessed_data(self, data: Dict[str, Any], dataset_name: str,
                              metadata: Optional[Dict] = None):
        """
        Save preprocessed data with metadata.

        Args:
            data: Dictionary containing preprocessed data
            dataset_name: Name for the dataset
            metadata: Optional metadata to save with the data
        """
        timestamp = datetime.now().isoformat()

        # Add metadata
        full_data = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'model_name': self.model_name
        }

        # Save as pickle
        file_path = self.preprocessed_dir / f'{dataset_name}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(full_data, f)

        self.logger.info(f"Saved preprocessed data: {file_path}")
        return file_path

    def load_preprocessed_data(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load preprocessed data.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Dictionary containing data and metadata, or None if not found
        """
        file_path = self.preprocessed_dir / f'{dataset_name}.pkl'

        if not file_path.exists():
            self.logger.warning(f"Preprocessed data not found: {file_path}")
            return None

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.logger.info(f"Loaded preprocessed data: {file_path}")
        return data

    def list_preprocessed_datasets(self) -> List[str]:
        """List all available preprocessed datasets."""
        datasets = [f.stem for f in self.preprocessed_dir.glob('*.pkl')]
        return sorted(datasets)

    # ============================================================================
    # Checkpoint Methods
    # ============================================================================

    def save_checkpoint(self, checkpoint_data: Dict[str, Any],
                       checkpoint_name: str = "model",
                       metadata: Optional[Dict] = None) -> Path:
        """
        Save model checkpoint with metadata.

        Args:
            checkpoint_data: Dictionary containing model state
            checkpoint_name: Name for the checkpoint
            metadata: Optional metadata (epoch, loss, etc.)

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().isoformat()

        # Add metadata
        full_checkpoint = {
            'checkpoint': checkpoint_data,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'model_name': self.model_name
        }

        # Save as .pt file
        file_path = self.checkpoints_dir / f'{checkpoint_name}.pt'
        torch.save(full_checkpoint, file_path)

        self.logger.info(f"Saved checkpoint: {file_path}")
        return file_path

    def load_checkpoint(self, checkpoint_name: str = "model") -> Optional[Dict[str, Any]]:
        """
        Load model checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint to load

        Returns:
            Dictionary containing checkpoint and metadata, or None if not found
        """
        file_path = self.checkpoints_dir / f'{checkpoint_name}.pt'

        if not file_path.exists():
            self.logger.warning(f"Checkpoint not found: {file_path}")
            return None

        # Use weights_only=False for compatibility with custom classes
        # Note: Only load checkpoints from trusted sources
        checkpoint = torch.load(file_path, weights_only=False)
        self.logger.info(f"Loaded checkpoint: {file_path}")
        return checkpoint

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        checkpoints = [f.stem for f in self.checkpoints_dir.glob('*.pt')]
        return sorted(checkpoints)

    def cleanup_checkpoints(self, keep_latest: int = 5):
        """
        Clean up old checkpoints, keeping only the latest N.

        Args:
            keep_latest: Number of latest checkpoints to keep
        """
        checkpoints = sorted(
            self.checkpoints_dir.glob('*.pt'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for checkpoint in checkpoints[keep_latest:]:
            checkpoint.unlink()
            self.logger.info(f"Removed old checkpoint: {checkpoint}")

    # ============================================================================
    # Sample Storage Methods
    # ============================================================================

    def save_samples(self, samples: Union[pd.DataFrame, np.ndarray, torch.Tensor],
                    sample_name: str = "synthetic",
                    format: str = "csv",
                    metadata: Optional[Dict] = None) -> Path:
        """
        Save generated samples.

        Args:
            samples: Generated samples (DataFrame, numpy array, or tensor)
            sample_name: Name for the samples
            format: Output format ('csv', 'pickle', 'parquet')
            metadata: Optional metadata (num_samples, generation_params, etc.)

        Returns:
            Path to saved samples
        """
        # Convert to DataFrame if needed
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        if isinstance(samples, np.ndarray):
            samples = pd.DataFrame(samples)

        # Save metadata separately
        if metadata:
            metadata_path = self.samples_dir / f'{sample_name}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump({
                    **metadata,
                    'timestamp': datetime.now().isoformat(),
                    'model_name': self.model_name,
                    'num_samples': len(samples)
                }, f, indent=2)

        # Save samples in requested format
        if format == 'csv':
            file_path = self.samples_dir / f'{sample_name}.csv'
            samples.to_csv(file_path, index=False)
        elif format == 'pickle':
            file_path = self.samples_dir / f'{sample_name}.pkl'
            samples.to_pickle(file_path)
        elif format == 'parquet':
            file_path = self.samples_dir / f'{sample_name}.parquet'
            samples.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Saved samples: {file_path}")
        return file_path

    def load_samples(self, sample_name: str = "synthetic",
                    format: str = "csv") -> Optional[pd.DataFrame]:
        """
        Load generated samples.

        Args:
            sample_name: Name of the samples to load
            format: Format of the samples ('csv', 'pickle', 'parquet')

        Returns:
            DataFrame containing samples, or None if not found
        """
        if format == 'csv':
            file_path = self.samples_dir / f'{sample_name}.csv'
            if file_path.exists():
                return pd.read_csv(file_path)
        elif format == 'pickle':
            file_path = self.samples_dir / f'{sample_name}.pkl'
            if file_path.exists():
                return pd.read_pickle(file_path)
        elif format == 'parquet':
            file_path = self.samples_dir / f'{sample_name}.parquet'
            if file_path.exists():
                return pd.read_parquet(file_path)

        self.logger.warning(f"Samples not found: {sample_name}.{format}")
        return None

    def list_samples(self) -> List[str]:
        """List all available sample sets."""
        samples = []
        for ext in ['*.csv', '*.pkl', '*.parquet']:
            samples.extend([f.stem for f in self.samples_dir.glob(ext)])
        return sorted(set(samples))

    # ============================================================================
    # Temporary File Methods
    # ============================================================================

    def get_temp_dir(self) -> Path:
        """Get path to temporary directory for this model."""
        return self.temp_dir

    def cleanup_temp_files(self):
        """Remove all temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Cleaned temp directory: {self.temp_dir}")

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage."""
        def get_dir_size(path: Path) -> int:
            """Get total size of directory in bytes."""
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        return {
            'model_name': self.model_name,
            'base_directory': str(self.base_data_dir),
            'model_directory': str(self.model_dir),
            'storage': {
                'preprocessed': {
                    'count': len(self.list_preprocessed_datasets()),
                    'size_mb': get_dir_size(self.preprocessed_dir) / (1024 * 1024)
                },
                'checkpoints': {
                    'count': len(self.list_checkpoints()),
                    'size_mb': get_dir_size(self.checkpoints_dir) / (1024 * 1024)
                },
                'samples': {
                    'count': len(self.list_samples()),
                    'size_mb': get_dir_size(self.samples_dir) / (1024 * 1024)
                },
                'temp': {
                    'size_mb': get_dir_size(self.temp_dir) / (1024 * 1024)
                }
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"DataManager(model='{self.model_name}', base_dir='{self.base_data_dir}')"