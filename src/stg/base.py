import torch
import threading


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
  def __init__(self, data_info, checkpoint_interval_seconds=None, epochs=None, messageSender=None, **kwargs):
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
  
  def train(
        self,
        train_data
    ):
    """Train a synthesizer:
        Args:
            train_data:
                A tensor dataloader object containing preprocessed training data, in numerical format suitable for synthesizer processing. 
        Returns:
            No return value.
    """
    self.start_threading()
    self.set_device()

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
        else:
            self.device = device
            
  def init_model(self, train_data):
    """Initialize attributes of the synthesizer"""
    pass
        
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

      self.messageSender.reportLoss(current_loss,remaining_epochs,estimated_remaining_time,message,self.passed_training_time)
      
      # Start another timer if necessary
      if self.current_epoch <= self._epochs:  # or whatever condition you want to stop the updates
          self.timer = threading.Timer(self.checkpoint_interval_seconds, self.update_frontend)
          self.timer.start()    