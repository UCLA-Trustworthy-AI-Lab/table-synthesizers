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
    data_info=None,
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
    BaseSynthesizer.__init__(self, data_info, checkpoint_interval_seconds, **kwarg)
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

    # Will be set when data_info is available (either from init or during training)
    self.num_cols = []
    self.ord_cols = []
    
    if self.data_info is not None:
        self._setup_column_info()

    self.diffusion = None
    self.ema_model = None
    self.empirical_class_dist = None
    
  def _setup_column_info(self):
    """Setup column information from data_info"""
    if self.data_info is not None:
        meta = self.data_info['transform_info']
        self.num_cols = [c for c in meta if meta[c]['original_dtype'] in ['continuous','ordinal', 'datetime','numerical']]
        self.ord_cols = [c for c in meta if meta[c]['original_dtype'] in ['ordinal']]
  
  def _custom_encode_dataframe(self, df):
    """TabDDPM-specific encoding that uses Gaussian Quantile Transform for numerical columns"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    
    df = df.copy()
    encoded_df = pd.DataFrame()
    data_info = {'transform_info': {}, 'encoded_width': 0, 'original_size': len(df)}
    
    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    start_idx = 0
    
    # Process numerical columns with Gaussian Quantile Transform (better for TabDDPM)
    for col in numerical_cols:
        # QuantileTransformer passes NaN straight through unchanged, which
        # then poisons the whole training loss (GLoss becomes NaN from the
        # first NaN-containing batch onward) and causes FoundNANsError at
        # sampling time. Median-impute before transforming, matching
        # sklearn's own SimpleImputer default strategy for numerical data.
        col_values = df[[col]]
        if col_values[col].isna().any():
            col_values = col_values.fillna(col_values[col].median())

        # Use QuantileTransformer with normal output distribution (better for diffusion models)
        scaler = QuantileTransformer(output_distribution='normal', random_state=self.seed)
        scaled_data = scaler.fit_transform(col_values)
        encoded_df[f'{col}_gqt'] = scaled_data.flatten()
        
        self.encoders[col] = {'type': 'quantile', 'encoder': scaler}
        
        data_info['transform_info'][col] = {
            'original_dtype': 'continuous',
            'start_idx': start_idx,
            'end_idx': start_idx + 1,
            'transformed_dtypes': {f'{col}_gqt': 'continuous'},
            'empirical_dist': []
        }
        start_idx += 1
    
    # Process categorical columns with Label Encoding (TabDDPM expects Label encoded data)
    for col in categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoded_data = encoder.fit_transform(df[col])
        
        # Convert to float for consistency
        encoded_df[f'{col}_le'] = encoded_data.astype(float)
        
        self.encoders[col] = {'type': 'label', 'encoder': encoder}

        # Calculate empirical distribution. LabelEncoder assigns NaN/None its
        # own class (encoder.classes_ includes it), but value_counts() drops
        # NaN by default -- using that directly undercounts num_classes by
        # one and misaligns per-index probabilities with the label indices,
        # causing an out-of-range embedding lookup during training whenever
        # the column actually contains missing values. Count with NaN
        # included and re-index to encoder.classes_'s order explicitly, so
        # length and per-index alignment both match the label encoding.
        counts = df[col].value_counts(normalize=True, dropna=False)
        empirical_dist = [float(counts.get(cls, 0.0)) for cls in encoder.classes_]
        
        data_info['transform_info'][col] = {
            'original_dtype': 'categorical',
            'start_idx': start_idx,
            'end_idx': start_idx + 1,
            'transformed_dtypes': {f'{col}_le': 'categorical'},
            'empirical_dist': empirical_dist
        }
        start_idx += 1
    
    data_info['encoded_width'] = encoded_df.shape[1]
    self.encoded_data = encoded_df
    self.feature_names = encoded_df.columns.tolist()
    
    return encoded_df, data_info
  
  def decode_samples(self, samples):
    """Decode TabDDPM samples back to original DataFrame format"""
    import torch
    import pandas as pd
    import numpy as np
    
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    # Convert to DataFrame with encoded feature names
    encoded_df = pd.DataFrame(samples, columns=self.feature_names)
    decoded_df = pd.DataFrame()
    
    # Reverse the encoding process
    for original_col, encoder_info in self.encoders.items():
        if encoder_info['type'] == 'quantile':
            # Find the GQT column
            gqt_col = f'{original_col}_gqt'
            if gqt_col in encoded_df.columns:
                # Create a DataFrame with the original column name for the scaler
                temp_df = pd.DataFrame({original_col: encoded_df[gqt_col]})
                decoded_values = encoder_info['encoder'].inverse_transform(temp_df).flatten()
                decoded_df[original_col] = decoded_values
        
        elif encoder_info['type'] == 'label':
            # Find the label encoded column
            le_col = f'{original_col}_le'
            if le_col in encoded_df.columns:
                # Round to integers and clip to valid range
                encoded_values = encoded_df[le_col].round().astype(int)
                n_classes = len(encoder_info['encoder'].classes_)
                encoded_values = np.clip(encoded_values, 0, n_classes - 1)
                
                decoded_values = encoder_info['encoder'].inverse_transform(encoded_values)
                decoded_df[original_col] = decoded_values
    
    return decoded_df

  def fit(self, data, batch_size=32):
    """Public fit method that calls the base class train method

    Args:
        data: pandas DataFrame or torch DataLoader
        batch_size: batch size for DataLoader creation when input is DataFrame
    """
    self.train(data, batch_size)

  def sample(self, n, return_dataframe=False):
    """Public sample method that calls the base class generate method

    Args:
        n: number of samples to generate
        return_dataframe: whether to return DataFrame (True) or tensor (False)
    """
    if return_dataframe:
        # Generate tensor samples and decode to DataFrame
        samples_tensor = self.generate(n)
        return self.decode_samples(samples_tensor)
    else:
        # Return raw tensor samples
        return self.generate(n)

  def _train(self, train_loader):
    # Setup column info if not already done (for DataFrame input)
    if not self.num_cols and not self.ord_cols:
        self._setup_column_info()

    if self.data_info is not None:
        columns = list(self.data_info['transform_info'])
        # Mirrors scripts/train.py's own auto-detection: defaults to the
        # last column when no explicit target was given.
        effective_target = self.target if self.target is not None else (columns[-1] if columns else None)
        feature_cols = [c for c in columns if c != effective_target]
        if effective_target is not None and not feature_cols:
            raise ValueError(
                f"TabDDPM requires at least one non-target feature column to train on; "
                f"got a dataset where '{effective_target}' is the only column. TabDDPM is a "
                f"conditional synthesizer -- it generates feature columns conditioned on "
                f"the target, so there is nothing for it to learn with zero feature columns."
            )

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