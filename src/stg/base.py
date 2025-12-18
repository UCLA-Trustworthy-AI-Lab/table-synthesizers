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


class BaseSynthesizer:
  """
    The parent class for all synthetic models. This class provides a template for defining custom synthesizers. 
    Subclasses should override the following methods as needed: __init__(), train(), generate(), get_state(), 
    load_state(), and optionally init_model().

    Attributes
    ----------
    model_loaded : bool
        Indicates whether the model is loaded.
    checkpoint_interval_seconds : int or None
        Interval in seconds to checkpoint during training.
    current_training_loss : float or None
        Current loss during training.
    current_epoch : int
        Current epoch in the training process.
    _epochs : int or None
        Total number of epochs for training.
    passed_training_time : float
        Total time passed in training.
    timer : threading.Timer or None
        Timer for periodic updates during training.
    messageSender : MessageSender or None
        Instance of MessageSender for reporting training progress.

    Methods
    -------
    __init__(checkpoint_interval_seconds=None, epochs=None, messageSender=None, **args, **kwargs)
        Initializes important parameters for the model with the option to use default values for unspecified parameters.

    train(train_data)
        Trains the synthesizer model.

    generate(n)
        Generates synthetic data samples.

    set_device(device=None)
        Sets the computation device for the model.

    init_model(train_data)
        Initializes attributes of the synthesizer.

    get_state()
        Retrieves necessary model states and parameters into a dictionary.

    load_state(checkpoint)
        Loads the model state from a checkpoint dictionary.

    start_threading()
        Initiates a timer for sending information to a client at regular intervals.

    stop_threading()
        Ends threading by cancelling the timer.

    update_frontend()
        Calculates and sends important training progress information to the client.
  """
  def __init__(self, data_info=None, checkpoint_interval_seconds=None, epochs=None, messageSender=None, seed: int = None,
               enable_data_manager: bool = True, enable_config_manager: bool = True,
               data_dir: Optional[str] = None, config_dir: Optional[str] = None, **kwargs):
    """
      Init important parameters for the model. You can enter parameters from the configuration json file. Parameters with no specification provided will use default values.

      Args:
          data_info: Information about data transformation
          checkpoint_interval_seconds: Interval for checkpointing during training
          epochs: Number of training epochs
          messageSender: Message sender for reporting progress
          seed: Random seed for reproducibility
          enable_data_manager: Enable DataManager for unified storage (default: True)
          enable_config_manager: Enable ConfigManager for configuration (default: True)
          data_dir: Base directory for data storage (default: 'data')
          config_dir: Directory for configuration files (default: 'config')
          **kwargs: Additional arguments
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
    """Train a synthesizer:
        Args:
            train_data:
                Either a pandas DataFrame (will be encoded automatically) or 
                a tensor dataloader object containing preprocessed training data.
            batch_size (int): Batch size for DataLoader creation when input is DataFrame.
        Returns:
            No return value.
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

  def _train(self, train_data):
    raise NotImplementedError("Training method need to be implemented by child synthesizers!")
    
  def generate(self, n, condition=None):
    """Sample data similar to the training data.
    Args:
        n (int):
            Number of samples to generate.
        condition (torch.dataloader): 
            A dataloader contains instance level condition to be generated based on.
    Returns:
        torch.tensor()
    """
        
    return self._generate(n, condition)

  def _generate(self, n, condition=None):
    raise NotImplementedError("Generating method need to be implemented by child synthesizers!")

  def set_device(self, device=None):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = self.device  # Also set _device for compatibility
        else:
            self.device = device
            self._device = device  # Also set _device for compatibility
  
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
    # Create custom dataset that returns tensors directly (not tuples)
    class SimpleTensorDataset(torch.utils.data.Dataset):
        def __init__(self, tensor):
            self.tensor = tensor
        def __len__(self):
            return self.tensor.size(0)
        def __getitem__(self, idx):
            return self.tensor[idx]
    
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
