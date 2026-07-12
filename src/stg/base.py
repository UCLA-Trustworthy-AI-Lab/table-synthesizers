import torch
import threading
import pandas as pd
import numpy as np
import logging
import random
import os
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

# Import managers as strict dependencies
from .data_manager import DataManager
from .config_manager import ConfigManager

# Import DataLoader for file loading.
# Prefer package import (`data_loader`) for installed/test environments,
# and fall back to local-src import for direct script execution.
try:
    from data_loader import DataLoader as FileDataLoader
except ImportError:  # pragma: no cover - fallback path
    try:
        from src.data_loader import DataLoader as FileDataLoader
    except ImportError:  # pragma: no cover - optional dependency
        FileDataLoader = None


class SimpleTensorDataset(torch.utils.data.Dataset):
  """Pickle-safe dataset wrapper that returns tensor rows directly."""

  def __init__(self, tensor):
    self.tensor = tensor

  def __len__(self):
    return self.tensor.size(0)

  def __getitem__(self, idx):
    return self.tensor[idx]


class BaseSynthesizer:
  """Abstract base class for all synthetic tabular data generators.

  Provides a unified interface for training and generating synthetic data,
  including automatic DataFrame encoding, device management, seeding,
  DataManager/ConfigManager integration, and optional frontend progress reporting.

  Subclasses must implement:
      - :meth:`_train`: core training logic, receives a ``torch.DataLoader``
      - :meth:`_generate`: core generation logic, returns a ``torch.Tensor``

  Subclasses may optionally override:
      - :meth:`get_state` / :meth:`load_state`: for checkpoint serialization
      - :meth:`_custom_encode_dataframe`: to replace the default MinMax+OneHot encoding
      - :meth:`apply_config`: to apply model-specific configuration keys

  Attributes:
      model_loaded (bool): Whether the model has been loaded from a checkpoint.
      checkpoint_interval_seconds (int | None): Seconds between frontend progress updates.
      current_training_loss (float | list | None): Most recent training loss.
      current_epoch (int): Current training epoch (0-indexed).
      _epochs (int | None): Total epochs to train.
      passed_training_time (float): Cumulative training time reported to frontend (seconds).
      timer (threading.Timer | None): Active progress-reporting timer.
      messageSender: Object with a ``reportLoss()`` method for frontend progress updates.
      data_info (dict | None): Column metadata produced by the encoder.
      encoders (dict): Per-column encoder objects (MinMaxScaler / OneHotEncoder).
      feature_names (list[str]): Encoded column names after encoding.
      device (torch.device): Active compute device set by :meth:`set_device`.
      data_manager (DataManager | None): Handles checkpoint and sample I/O.
      config_manager (ConfigManager | None): Handles JSON config loading.

  Example:
      >>> class MyModel(BaseSynthesizer):
      ...     def _train(self, train_data):
      ...         pass  # fit your model on the DataLoader
      ...     def _generate(self, n, condition=None):
      ...         return torch.randn(n, 10)
      ...
      >>> model = MyModel(epochs=100, seed=42)
      >>> model.train(df)                           # DataFrame → encoded → DataLoader → _train()
      >>> samples = model.generate(100)             # calls _generate(), returns torch.Tensor
      >>> df_out = model.sample(100, return_dataframe=True)
  """
  def __init__(self, data_info=None, checkpoint_interval_seconds=None, epochs=None, messageSender=None, seed: int = None,
               enable_data_manager: bool = True, enable_config_manager: bool = True,
               data_dir: Optional[str] = None, config_dir: Optional[str] = None, **kwargs):
    """Initialize the synthesizer.

    Args:
        data_info (dict | None): Pre-computed column metadata (``transform_info``,
            ``encoded_width``, etc.). When ``None``, it is inferred automatically
            during the first :meth:`train` call on a DataFrame.
        checkpoint_interval_seconds (int | None): If set, a background thread fires
            every this many seconds and calls :meth:`update_frontend`.
        epochs (int | None): Total training epochs. Used for progress estimation.
        messageSender: Object exposing ``reportLoss(loss, remaining_epochs,
            remaining_time, message, elapsed_time)`` for frontend callbacks.
        seed (int | None): Global random seed applied to ``random``, ``numpy``,
            and ``torch`` before training starts.
        enable_data_manager (bool): Create a :class:`~stg.data_manager.DataManager`
            for checkpoint and sample storage. Default ``True``.
        enable_config_manager (bool): Create a :class:`~stg.config_manager.ConfigManager`
            for JSON config loading. Default ``True``.
        data_dir (str | None): Root directory for DataManager storage.
            Defaults to ``'data'``.
        config_dir (str | None): Directory for ConfigManager JSON files.
            Defaults to ``'config'``.
        **kwargs: Forwarded to subclass ``__init__`` implementations.
    """
    self.model_loaded = False
    self.checkpoint_interval_seconds = checkpoint_interval_seconds
    self.current_training_loss = None
    self.current_epoch = 0
    self._epochs = epochs
    self.passed_training_time = 0
    self.timer =None
    self.messageSender = messageSender
    self.data_info = data_info
    # Reproducibility
    self._seed = seed

    # Encoding components
    self.encoders = {}
    self.column_info = {}
    self.encoded_data = None
    self.feature_names = []
    # Logger
    self._logger = logging.getLogger(__name__)

    # Initialize managers
    self.data_manager = None
    self.config_manager = None

    if enable_data_manager:
        model_name = self.__class__.__name__
        self.data_manager = DataManager(model_name, base_data_dir=data_dir)
        self._logger.info(f"DataManager initialized for {model_name}")

    if enable_config_manager:
        self.config_manager = ConfigManager(config_dir=config_dir)
        self._logger.info("ConfigManager initialized")
  
  def train(
        self,
        train_data,
        batch_size=32
    ):
    """Train the synthesizer.

    Handles device setup, seeding, DataFrame-to-DataLoader conversion, and
    frontend progress threading before delegating to :meth:`_train`.

    Args:
        train_data (pd.DataFrame | torch.utils.data.DataLoader): Training data.
            - **DataFrame**: columns are auto-encoded (MinMax for numeric,
              OneHot for categorical) and wrapped in a ``DataLoader``.
            - **DataLoader**: passed directly to :meth:`_train` (caller is
              responsible for preprocessing).
        batch_size (int): Batch size used only when ``train_data`` is a DataFrame.
            Ignored for DataLoader input. Default ``32``.

    Raises:
        ValueError: If ``train_data`` is neither a DataFrame nor a DataLoader.
    """
    self.start_threading()
    # Set seed deterministically if provided
    self.set_seed(self._seed)
    self.set_device()

    # Handle different input types
    if isinstance(train_data, pd.DataFrame):
        train_data = self._prepare_dataloader_from_dataframe(train_data, batch_size)
    elif not hasattr(train_data, '__iter__') or not hasattr(train_data, 'dataset'):
        raise ValueError("train_data must be either a pandas DataFrame or a torch DataLoader")

    self._train(train_data)

    self.stop_threading()

  def train_from_csv(self, file_path: str, optimize_memory: bool = False, batch_size: int = 32):
    """
    Train synthesizer directly from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        optimize_memory: If True, apply memory optimization (downcasting, categorical conversion)
        batch_size: Batch size for DataLoader creation
    
    Raises:
        ImportError: If DataLoader is not available
        FileNotFoundError: If the file does not exist
    """
    if FileDataLoader is None:
        raise ImportError("DataLoader not available. Please ensure data_loader package is installed.")
    
    loader = FileDataLoader()
    df = loader.load(file_path, optimize_memory=optimize_memory)
    self.train(df, batch_size=batch_size)

  def train_from_parquet(self, file_path: str, optimize_memory: bool = False, batch_size: int = 32):
    """
    Train synthesizer directly from a Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        optimize_memory: If True, apply memory optimization (downcasting, categorical conversion)
        batch_size: Batch size for DataLoader creation
    
    Raises:
        ImportError: If DataLoader is not available
        FileNotFoundError: If the file does not exist
    """
    if FileDataLoader is None:
        raise ImportError("DataLoader not available. Please ensure data_loader package is installed.")
    
    loader = FileDataLoader()
    df = loader.load(file_path, optimize_memory=optimize_memory)
    self.train(df, batch_size=batch_size)

  def _train(self, train_data):
    raise NotImplementedError("Training method need to be implemented by child synthesizers!")
    
  def generate(self, n, condition=None):
    """Generate synthetic samples.

    Delegates directly to :meth:`_generate`. Use :meth:`sample` for an
    sklearn-style interface that can return a decoded DataFrame.

    Args:
        n (int): Number of samples to generate.
        condition (torch.utils.data.DataLoader | None): Optional instance-level
            conditioning dataloader. Length must match ``n``.

    Returns:
        torch.Tensor: Raw generated samples in encoded space.
    """
        
    return self._generate(n, condition)

  def fit(self, data, batch_size=32):
    """sklearn-style interface for training.
    
    This is a convenience method that delegates to train().
    Provided for sklearn compatibility.
    
    Args:
        data: Either a pandas DataFrame or a torch DataLoader
        batch_size: Batch size for DataLoader creation when input is DataFrame
    """
    self.train(data, batch_size=batch_size)

  def sample(self, n_samples, return_dataframe=False):
    """sklearn-style interface for generation.
    
    This is a convenience method that delegates to generate() and optionally
    decodes the output to a DataFrame.
    
    Args:
        n_samples: Number of samples to generate
        return_dataframe: If True, decode samples to DataFrame format
    
    Returns:
        torch.Tensor or pd.DataFrame: Generated samples
    """
    synth_data = self.generate(n_samples)
    if return_dataframe and hasattr(self, 'decode_samples'):
        return self.decode_samples(synth_data)
    return synth_data

  def _generate(self, n, condition=None):
    raise NotImplementedError("Generating method need to be implemented by child synthesizers!")

  def set_device(self, device=None):
        """
        Set the device to be used for training ('cuda', 'mps', 'cpu', or 'auto').

        Args:
            device (str or torch.device): Device to use. Options:
                - 'cuda': Use NVIDIA GPU (requires CUDA)
                - 'mps': Use Apple Silicon GPU (requires macOS with M1/M2/M3)
                - 'cpu': Use CPU
                - 'auto': Automatically detect best available device
                - None: Same as 'auto'
                - torch.device: Directly provide torch device object

        Note:
            For GPU training, ensure PyTorch is installed with CUDA support:
            - CUDA 13.0+ for Blackwell (SM 12.1+): pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
            - CUDA 12.4 for most GPUs: pip install torch --index-url https://download.pytorch.org/whl/cu124
        """
        if device is None or device == 'auto':
            # Auto-detect best device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logging.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logging.info("Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device("cpu")
                logging.info("Using CPU (GPU not available)")
        elif isinstance(device, str):
            if device == 'cuda':
                if not torch.cuda.is_available():
                    logging.warning("CUDA requested but not available. Falling back to CPU.")
                    self.device = torch.device("cpu")
                else:
                    self.device = torch.device("cuda")
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logging.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            elif device == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    logging.warning("MPS requested but not available. Falling back to CPU.")
                    self.device = torch.device("cpu")
                else:
                    self.device = torch.device("mps")
                    logging.info("Using Apple Silicon GPU (MPS)")
            elif device == 'cpu':
                self.device = torch.device("cpu")
                logging.info("Using CPU")
            else:
                logging.warning(f"Unknown device '{device}'. Using auto-detection.")
                self.set_device('auto')
                return
        else:
            # torch.device object provided
            self.device = device
            logging.info(f"Using device: {device}")

        # Set _device for backward compatibility
        self._device = self.device

  def get_optimal_batch_size(self, dataset_size, default_batch_size=128):
        """
        Calculate optimal batch size based on GPU memory and dataset size.

        Args:
            dataset_size (int): Number of samples in the dataset
            default_batch_size (int): Default batch size if GPU not available

        Returns:
            int: Recommended batch size

        Note:
            This is a heuristic. Actual optimal batch size depends on model architecture.
        """
        if not hasattr(self, 'device'):
            self.set_device('auto')

        if self.device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            # Heuristic: adjust batch size based on GPU memory
            if gpu_memory_gb >= 80:  # High-end GPUs (A100, H100, GB10, etc.)
                batch_size = min(512, dataset_size // 2)
            elif gpu_memory_gb >= 40:  # Mid-high GPUs (A6000, RTX 6000, etc.)
                batch_size = min(256, dataset_size // 4)
            elif gpu_memory_gb >= 16:  # Consumer high-end (RTX 4090, RTX 3090, etc.)
                batch_size = min(128, dataset_size // 8)
            elif gpu_memory_gb >= 8:   # Consumer mid-range (RTX 4070, RTX 3070, etc.)
                batch_size = min(64, dataset_size // 10)
            else:                       # Entry-level GPUs
                batch_size = min(32, dataset_size // 10)

            logging.info(f"Recommended batch size for {gpu_memory_gb:.1f}GB GPU: {batch_size}")
            return batch_size
        else:
            # CPU or MPS - use conservative batch size
            return min(default_batch_size, dataset_size // 10)
  
  def set_seed(self, seed: int = None):
    """Set random seeds for reproducibility across torch, numpy, and python's random."""
    if seed is None:
        return
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

  
  def init_model(self, train_data):
    """Initialize attributes of the synthesizer"""
    pass
  
  def _prepare_dataloader_from_dataframe(self, df, batch_size=32):
    """Convert pandas DataFrame to encoded tensor DataLoader"""
    # Encode the DataFrame
    encoded_df, data_info = self._encode_dataframe(df)
    
    # Store data_info if not already set
    if self.data_info is None:
        self.data_info = data_info
    
    # Convert to tensor
    tensor_data = torch.tensor(encoded_df.values, dtype=torch.float32)
    
    # Create dataset and dataloader
    # Note: dataset class is module-level to keep checkpoints pickle-safe.
    dataset = SimpleTensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
  
  def _encode_dataframe(self, df):
    """Encode DataFrame using default or custom encoding methods"""
    if hasattr(self, '_custom_encode_dataframe'):
        return self._custom_encode_dataframe(df)
    else:
        return self._default_encode_dataframe(df)
  
  def _default_encode_dataframe(self, df):
    """Default encoding: MinMax for numerical, OneHot for categorical"""
    df = df.copy()
    encoded_df = pd.DataFrame()
    data_info = {'transform_info': {}, 'encoded_width': 0, 'original_size': len(df)}
    
    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    start_idx = 0
    
    # Process numerical columns
    for col in numerical_cols:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[[col]])
        encoded_df[f'{col}_scaled'] = scaled_data.flatten()
        
        self.encoders[col] = {'type': 'minmax', 'encoder': scaler}
        
        data_info['transform_info'][col] = {
            'original_dtype': 'continuous',
            'start_idx': start_idx,
            'end_idx': start_idx + 1,
            'transformed_dtypes': {f'{col}_scaled': 'continuous'},
            'empirical_dist': []
        }
        start_idx += 1
    
    # Process categorical columns  
    for col in categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[[col]])
        
        # Get feature names
        feature_names = encoder.get_feature_names_out([col])
        for i, feature_name in enumerate(feature_names):
            encoded_df[feature_name] = encoded_data[:, i]
        
        self.encoders[col] = {'type': 'onehot', 'encoder': encoder}
        
        # Calculate empirical distribution
        empirical_dist = df[col].value_counts(normalize=True).values.tolist()
        
        data_info['transform_info'][col] = {
            'original_dtype': 'categorical',
            'start_idx': start_idx,
            'end_idx': start_idx + len(feature_names),
            'transformed_dtypes': {fn: 'binary' for fn in feature_names},
            'empirical_dist': empirical_dist
        }
        start_idx += len(feature_names)
    
    data_info['encoded_width'] = encoded_df.shape[1]
    self.encoded_data = encoded_df
    self.feature_names = encoded_df.columns.tolist()
    
    return encoded_df, data_info
  
  def decode_samples(self, samples):
    """Decode generated samples back to original DataFrame format"""
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    # Convert to DataFrame with encoded feature names
    encoded_df = pd.DataFrame(samples, columns=self.feature_names)
    decoded_df = pd.DataFrame()
    
    # Reverse the encoding process
    for original_col, encoder_info in self.encoders.items():
        if encoder_info['type'] == 'minmax':
            # Find the scaled column
            scaled_col = f'{original_col}_scaled'
            if scaled_col in encoded_df.columns:
                decoded_values = encoder_info['encoder'].inverse_transform(
                    encoded_df[[scaled_col]]
                ).flatten()
                decoded_df[original_col] = decoded_values
        
        elif encoder_info['type'] == 'onehot':
            # Find all encoded columns for this original column
            encoder = encoder_info['encoder']
            feature_names = encoder.get_feature_names_out([original_col])
            
            if all(fn in encoded_df.columns for fn in feature_names):
                encoded_cols_data = encoded_df[feature_names].values
                decoded_values = encoder.inverse_transform(encoded_cols_data).flatten()
                decoded_df[original_col] = decoded_values
    
    return decoded_df
        
  def get_state(self):
    """Write necessary model states and parameters into one dictionary. """
    pass


  def load_state(self, checkpoint):
    """Load state from a checkpoint dictionary"""
    pass

  # ============================================================================
  # DataManager Integration Methods
  # ============================================================================

  def save_checkpoint_to_manager(self, checkpoint_name: str = None, metadata: Optional[Dict] = None):
    """
    Save model checkpoint using DataManager.

    Args:
        checkpoint_name: Name for the checkpoint (default: 'epoch_{current_epoch}')
        metadata: Optional metadata to save with checkpoint

    Returns:
        Path to saved checkpoint or None if DataManager not enabled
    """
    if self.data_manager is None:
        self._logger.warning("DataManager not enabled, checkpoint not saved")
        return None

    # Get model state
    checkpoint_data = self.get_state()

    # Auto-generate checkpoint name if not provided
    if checkpoint_name is None:
        checkpoint_name = f"epoch_{self.current_epoch}"

    # Add automatic metadata
    auto_metadata = {
        'epoch': self.current_epoch,
        'model_type': self.__class__.__name__,
        'seed': self._seed
    }
    if self.current_training_loss is not None:
        auto_metadata['loss'] = float(self.current_training_loss)

    # Merge with user-provided metadata
    if metadata:
        auto_metadata.update(metadata)

    # Save checkpoint
    checkpoint_path = self.data_manager.save_checkpoint(
        checkpoint_data,
        checkpoint_name,
        metadata=auto_metadata
    )

    self._logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

  def load_checkpoint_from_manager(self, checkpoint_name: str = 'final_model'):
    """
    Load model checkpoint using DataManager.

    Args:
        checkpoint_name: Name of the checkpoint to load

    Returns:
        True if loaded successfully, False otherwise
    """
    if self.data_manager is None:
        self._logger.warning("DataManager not enabled, cannot load checkpoint")
        return False

    # Get the file path to the checkpoint
    checkpoint_path = self.data_manager.checkpoints_dir / f'{checkpoint_name}.pt'

    if not checkpoint_path.exists():
        self._logger.warning(f"Checkpoint '{checkpoint_name}' not found at {checkpoint_path}")
        return False

    # Load the checkpoint file
    checkpoint_data = torch.load(checkpoint_path, weights_only=False)

    # Pass the checkpoint data directly to load_state
    # The checkpoint dict contains both the model state and metadata
    if 'checkpoint' in checkpoint_data:
        # New format: wrapped with metadata
        self.load_state(checkpoint_data['checkpoint'])
        metadata = checkpoint_data.get('metadata', {})
    else:
        # Old format: direct state dict
        self.load_state(checkpoint_data)
        metadata = {}

    # Restore epoch and metadata if available
    if 'epoch' in metadata:
        self.current_epoch = metadata['epoch']

    self._logger.info(f"Checkpoint loaded: {checkpoint_name}, metadata: {metadata}")
    return True

  def save_samples_to_manager(self, samples, sample_name: str = None,
                               format: str = 'csv', metadata: Optional[Dict] = None):
    """
    Save generated samples using DataManager.

    Args:
        samples: Generated samples (DataFrame, tensor, or array)
        sample_name: Name for the samples (default: 'samples_epoch_{current_epoch}')
        format: Output format ('csv', 'pickle', 'parquet')
        metadata: Optional metadata

    Returns:
        Path to saved samples or None if DataManager not enabled
    """
    if self.data_manager is None:
        self._logger.warning("DataManager not enabled, samples not saved")
        return None

    # Auto-generate sample name if not provided
    if sample_name is None:
        sample_name = f"samples_epoch_{self.current_epoch}"

    # Add automatic metadata
    auto_metadata = {
        'epoch': self.current_epoch,
        'model_type': self.__class__.__name__,
        'seed': self._seed
    }
    if metadata:
        auto_metadata.update(metadata)

    # Save samples
    samples_path = self.data_manager.save_samples(
        samples,
        sample_name,
        format=format,
        metadata=auto_metadata
    )

    self._logger.info(f"Samples saved: {samples_path}")
    return samples_path

  def save_preprocessed_data_to_manager(self, data_dict: Dict, dataset_name: str,
                                        metadata: Optional[Dict] = None):
    """
    Save preprocessed data using DataManager.

    Args:
        data_dict: Dictionary containing preprocessed data
        dataset_name: Name for the dataset
        metadata: Optional metadata

    Returns:
        Path to saved data or None if DataManager not enabled
    """
    if self.data_manager is None:
        self._logger.warning("DataManager not enabled, data not saved")
        return None

    data_path = self.data_manager.save_preprocessed_data(
        data_dict,
        dataset_name,
        metadata=metadata
    )

    self._logger.info(f"Preprocessed data saved: {data_path}")
    return data_path

  # ============================================================================
  # ConfigManager Integration Methods
  # ============================================================================

  def load_config_from_manager(self, profile: str = 'default'):
    """
    Load configuration for this model using ConfigManager.

    Args:
        profile: Configuration profile to load

    Returns:
        Configuration dictionary or None if ConfigManager not enabled
    """
    if self.config_manager is None:
        self._logger.warning("ConfigManager not enabled, cannot load config")
        return None

    model_name = self.__class__.__name__
    config = self.config_manager.load_config(model_name, profile=profile)

    self._logger.info(f"Config loaded for {model_name}, profile: {profile}")
    return config

  def apply_config(self, config: Dict):
    """
    Apply configuration to model parameters.

    Args:
        config: Configuration dictionary to apply

    Note:
        Subclasses should override this method to apply model-specific configs
    """
    # Apply training configuration
    if 'training' in config:
        if 'epochs' in config['training']:
            self._epochs = config['training']['epochs']
        # Subclasses can override to apply more parameters

    self._logger.info(f"Configuration applied: {config.keys()}")
    
  def start_threading(self):
    """
    It initiates a timer which sends info to a client every interval. 
    Interval length is defined by checkpoint_interval_seconds. 
    """
    
    if self.checkpoint_interval_seconds is not None:
      self._logger.debug("Progress thread started")
      self.timer = threading.Timer(self.checkpoint_interval_seconds, self.update_frontend)
      self.timer.start()
      
  def stop_threading(self):
    """
    This function ends threading by cancelling the timer and set the current_epoch to exceed the _epochs. 
    """
    # No need to do anything if checkpoint_interval_seconds is None.
    if self.checkpoint_interval_seconds is None:
       return
    self.current_epoch = self._epochs + 1
    # Let front end know that the training is finished
    self.update_frontend()
    if self.timer is not None:
      #print("Timer cancelled!")
      self.timer.cancel()
      #print("self.currenet_epoch ",self.current_epoch )
    

  def update_frontend(self):
      """
      This function calculates important training progresses and send to client: 
      Things to be passed:
      1, current_loss
      2, remaining_epochs
      3, estimated_remaining_time
      4, message
      And start a new timer if the training has not finished. 
      """
      # If training has not started, return a message saying the transformation is ongoing. 
      # Currently we have no mean of tracking transformation progress.
      if self.current_epoch == 0:
          current_loss = estimated_remaining_time = remaining_epochs = float('inf')
          message = "Transforming real dataset.."
      elif self.current_epoch == self._epochs + 1:
          current_loss = estimated_remaining_time = remaining_epochs = -1
          message = "Training completed!"
      else:
          # Note: for GAN, current_loss will be a list of two numbers since generator and discriminator has two different losses.
          # In that case the first number shall always be generator loss and second be discriminator loss. 
          # But for other models like transformer, current_loss will be just on number.
          current_loss = self.current_training_loss # Note: current_loss might be a list(for GAN)
          self.passed_training_time += self.checkpoint_interval_seconds
          estimated_time_per_epoch = self.passed_training_time / self.current_epoch
          remaining_epochs = (self._epochs - self.current_epoch)
          estimated_remaining_time = remaining_epochs * estimated_time_per_epoch
          message = f"Current training_epoch: {self.current_epoch}. Training loss: {current_loss}. Estimated remaining time: {estimated_remaining_time}."
      # Send info to frontend.
      if self.messageSender is not None:
          self.messageSender.reportLoss(current_loss,remaining_epochs,estimated_remaining_time,message,self.passed_training_time)
      
      # Start another timer if necessary
      if self.current_epoch <= self._epochs:  # or whatever condition you want to stop the updates
          self.timer = threading.Timer(self.checkpoint_interval_seconds, self.update_frontend)
          self.timer.start()    
