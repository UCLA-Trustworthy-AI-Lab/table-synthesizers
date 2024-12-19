from .scripts.train import train
from .scripts.sample import sample


from ..base import BaseSynthesizer


class TabDDPM(BaseSynthesizer):
  """
    A synthesizer for modeling tabular data using Gaussian diffusion for numerical variables and Multinomial diffusion 
    for categorical variables. The implementation is based on the work found at https://github.com/yandex-research/tab-ddpm.

    Methods
    -------
    __init__(meta=None, TabDDPMconfig=None, change_val=False, seed=42, cuda=True, training_steps=10000, checkpoint=None, checkpoint_interval_seconds=None, **kwargs)
        Initializes the TabDDPM synthesizer with the given parameters.

    train_data_to_np(real_data_path)
        Converts training data to NumPy format and saves it to a specified path.

    train(train_data, *ignore)
        Trains the TabDDPM model using the provided training data.

    generate(n)
        Generates n samples similar to the training data.

    get_state()
        Retrieves the current state of the synthesizer, including the training data and diffusion model.

    load_state(checkpoint)
        Loads the synthesizer state from a given checkpoint.

    Parameters
    ----------
    seed : int
        Seed for random number generation.
    cuda : bool
        Flag to use CUDA for computation.
    training_steps : int
        Number of steps to train the model.
    checkpoint : various types
        Checkpoint data for model initialization.
    checkpoint_interval_seconds : int or None
        Interval in seconds for checkpointing during training.
    **kwargs : dict
        Additional keyword arguments.
"""
  def __init__(self,
    data_info,
    target=None,
    steps = 10000,
    lr = 0.001,
    weight_decay = 1e-4,
    model_type = 'mlp',
    d_layers = [256, 256],
    dropout = 0.0,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    seed = 0,
    sample_batch_size = 2000,
    checkpoint_interval_seconds=None,**kwarg):
    BaseSynthesizer.__init__(self, checkpoint_interval_seconds)
    self.target = target
    self.steps = steps
    self.lr = lr
    self.weight_decay = weight_decay
    self.model_type = model_type
    self.model_params = {"rtdl_params":{"d_layers":d_layers, "dropout":dropout}}
    self.num_timesteps = num_timesteps
    self.gaussian_loss_type = gaussian_loss_type
    self.scheduler = scheduler
    self.seed = seed
    self.sample_batch_size = sample_batch_size

    self.data_info = data_info
    meta = data_info['transform_info']
    self.num_cols = [c for c in meta if meta[c]['original_dtype'] in ['continuous','ordinal', 'datetime','numerical']]
    self.ord_cols = [c for c in meta if meta[c]['original_dtype'] in ['ordinal']]
    #print("self.num_cols in TABDDPM:",self.num_cols)
    #print("data_info in tabpddpm init:", data_info)

    self.diffusion = None
    self.ema_model = None
    self.empirical_class_dist = None

  def _train(self, train_loader):
    self.diffusion_fn, self.ema_model, self.empirical_class_dist = train(train_loader,
    self.data_info,
    self.target,
    self.steps,
    self.lr,
    self.weight_decay,
    self.model_type,
    self.model_params,
    self.num_timesteps,
    self.gaussian_loss_type,
    self.scheduler,
    device = self.device,
    seed = self.seed)

  def _generate(self, n, condition=None):
    return sample(
    self.data_info,
    self.target,
    self.empirical_class_dist,
    self.sample_batch_size,
    n,
    self.model_type,
    self.model_params,
    self.diffusion_fn.state_dict(),
    self.num_timesteps,
    self.gaussian_loss_type,
    self.scheduler,
    device = self.device,
    seed = self.seed)
        
  def get_state(self):
        # Returns a dictionary of the current state of all attributes
        state = {attr: getattr(self, attr) for attr in self.__dict__}
        return state

  def load_state(self, state):
        # Sets the attributes based on the provided state dictionary
        for attr, value in state.items():
            setattr(self, attr, value)