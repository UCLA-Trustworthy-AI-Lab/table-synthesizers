import torch
import threading
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder


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
  def __init__(self, data_info=None, checkpoint_interval_seconds=None, epochs=None, messageSender=None, **kwargs):
    """
      Init important parameters for the model. You can enter parameters from the configuration json file. Parameters with no specification provided will use default values.
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
    
    # Encoding components
    self.encoders = {}
    self.column_info = {}
    self.encoded_data = None
    self.feature_names = []
  
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
    
  def start_threading(self):
    """
    It initiates a timer which sends info to a client every interval. 
    Interval length is defined by checkpoint_interval_seconds. 
    """
    
    if self.checkpoint_interval_seconds is not None:
      print("Thread started!")
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