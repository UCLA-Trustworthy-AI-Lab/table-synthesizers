from .base import BaseSynthesizer
from .identity import Identity
from .CTGAN import CTGAN
from .TabDDPM import TabDDPM
from .PATECTGAN import PATECTGAN
try:
    from .AIM import AIM
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("Warning: AIM not available due to missing dependencies")
from .TVAE import TVAE

# New synthesizers that only support DataFrame input
from .CART import CARTSynthesizer
from .DPCART import DPCARTSynthesizer
try:
    from .SMOTE import SMOTESynthesizer
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: SMOTE not available due to missing dependencies (imbalanced-learn)")

# New synthesizers from Gen_MIA experiments
try:
    from .BayesianNetwork import BayesianNetworkSynthesizer
    BAYESIANNETWORK_AVAILABLE = True
except (ImportError, AttributeError) as e:
    BAYESIANNETWORK_AVAILABLE = False
    print(f"Warning: BayesianNetwork not available due to dependencies: {str(e)}")

try:
    from .GREAT import GREATSynthesizer
    GREAT_AVAILABLE = True
except (ImportError, AttributeError) as e:
    GREAT_AVAILABLE = False
    print(f"Warning: GREAT not available due to dependencies: {str(e)}")

try:
    from .ARF import ARFSynthesizer
    ARF_AVAILABLE = True
except (ImportError, AttributeError) as e:
    ARF_AVAILABLE = False
    print(f"Warning: ARF not available due to dependencies: {str(e)}")

try:
    from .NFlow import NFlowSynthesizer
    NFLOW_AVAILABLE = True
except (ImportError, AttributeError) as e:
    NFLOW_AVAILABLE = False
    print(f"Warning: NFlow not available due to dependencies: {str(e)}")

try:
    from .AutoDiff import AutoDiffSynthesizer
    AUTODIFF_AVAILABLE = True
except ImportError:
    AUTODIFF_AVAILABLE = False
    print("Warning: AutoDiff not available due to missing dependencies")

try:
    from .TabSyn import TabSynSynthesizer
    TABSYN_AVAILABLE = True
except ImportError:
    TABSYN_AVAILABLE = False
    print("Warning: TabSyn not available due to missing dependencies")

import numpy as np
import torch

DEFAULT_MODELS = {"Identity":Identity,
                  "CTGAN":CTGAN,
                  "TabDDPM":TabDDPM,
                  "PATECTGAN":PATECTGAN,
                  "TVAE":TVAE,
                  "CART":CARTSynthesizer,
                  "DPCART":DPCARTSynthesizer}

if AIM_AVAILABLE:
    DEFAULT_MODELS["AIM"] = AIM

if SMOTE_AVAILABLE:
    DEFAULT_MODELS["SMOTE"] = SMOTESynthesizer

if BAYESIANNETWORK_AVAILABLE:
    DEFAULT_MODELS["BayesianNetwork"] = BayesianNetworkSynthesizer

if GREAT_AVAILABLE:
    DEFAULT_MODELS["GREAT"] = GREATSynthesizer

if ARF_AVAILABLE:
    DEFAULT_MODELS["ARF"] = ARFSynthesizer

if NFLOW_AVAILABLE:
    DEFAULT_MODELS["NFlow"] = NFlowSynthesizer

if AUTODIFF_AVAILABLE:
    DEFAULT_MODELS["AutoDiff"] = AutoDiffSynthesizer

if TABSYN_AVAILABLE:
    DEFAULT_MODELS["TabSyn"] = TabSynSynthesizer

VALID_DTYPES = set(['continuous', 'bounded_continuous', "ordinal", 'binary', "categorical", 'datetime', 'text', 'pii', 'index'])

class TableSynthesizer:
    """
        TableSynthesizer does the following: 
        1, takes a tensor dataset as input
        2, load column to tensor mapping
        3, initialize synthesizer with configurations
        4, pass tensor and mapping to selected synthesizer.
        5, generate synthetic tensor
        6, return synthetic tensors to preprocessors. 
    """

    _DEFAULT_MODELS = DEFAULT_MODELS
    _VALID_DTYPES = VALID_DTYPES
    def __init__(self, model, config=None, data_info=None, **kwarg):
        """
        An interface that allows the construction and selection of different synthesizers. It reduces the need for changing the code each time we want to run a different synthesizer. 

        Args:
            model (str or BaseSynthesizer): choice of synthesizer model name or instance.
            config (dict): training parameters. Optional when using DataFrame input.
            data_info (dict): metadata of transformed table. Optional when using DataFrame input.
        """
        
        if config is None:
            config = {}
            
        if data_info is not None:
            self.validate_data_info(data_info)
        
        if isinstance(model, str) and model in DEFAULT_MODELS:
            self.model = DEFAULT_MODELS[model](data_info=data_info,**config)
        elif isinstance(model, BaseSynthesizer):
            self.model = model
        else:
            raise ValueError(f"model provided must be a BaseSynthesizer instance, or a string in the following default models: {self.get_registered_models()}. Got: {model}. ")
        
    @classmethod
    def register_model(cls, new_models):
        """
            This methods adds new models to _DEFAULT_MODELS so that they can be selected by passing in model name. Custom models by users must be first registered.
        """
        assert isinstance(new_models, dict), "New models must be a dictionary with model name/model class pairs!"
        for md in new_models.values():
            assert issubclass(md, BaseSynthesizer), f"The model being registered must be a BaseSynthesizer instance! Received {type(md)}"
        cls._DEFAULT_MODELS.update(new_models)

    @classmethod
    def get_registered_models(cls):
        return cls._DEFAULT_MODELS

    def validate_data_info(self, data_info):
        """
        This function confirms data_info follows the format: 
        {transform_info: {column_name: {original_dtype, start_idx, end_idx, transformed_dtypes, empirical_dist}},
        encoded_width: int}
        transformed_dtypes = {transformed_column_with_surfix: transformed_column_dtypes}
        
        Parameters:
        data_info (dict): The data_info dictionary to validate
        
        Returns:
        bool: True if the format is correct, False otherwise
        
        Raises:
        ValueError: If any validation check fails.
        """
        # Check if 'transform_info' and 'encoded_width' are in data_info
        if 'transform_info' not in data_info or 'encoded_width' not in data_info:
            raise ValueError("data_info must contain 'transform_info' and 'encoded_width' keys.")

        # Check if 'encoded_width' is an integer
        if not isinstance(data_info['encoded_width'], int):
            raise ValueError(f"Encoded width {data_info['encoded_width']} is not an integer.")

        encoded_width = data_info['encoded_width']

        # Iterate through each entry in the transform_info dictionary
        for column_name, info in data_info['transform_info'].items():
            # Check if column_name is a string
            if not isinstance(column_name, str):
                raise ValueError(f"Column name {column_name} is not a string.")

            # Check if info is a dictionary
            if not isinstance(info, dict):
                raise ValueError(f"Info for column {column_name} is not a dictionary.")

            # Check required keys and their types in the info dictionary
            required_keys = ['original_dtype', 'start_idx', 'end_idx', 'transformed_dtypes', "empirical_dist"]
            for key in required_keys:
                if key not in info:
                    raise ValueError(f"Key {key} is missing in the info dictionary for column {column_name}.")

                if key == 'original_dtype':
                    if info[key] not in self._VALID_DTYPES:
                        raise ValueError(
                            f"Dtype {info[key]} given for original column {column_name} is not valid. "
                            f"Acceptable dtypes: {self._VALID_DTYPES}."
                        )
                elif key == 'transformed_dtypes':
                    if not isinstance(info[key], dict):
                        raise ValueError(f"transformed_dtypes for column {column_name} should be a dictionary.")
                    for transformed_column, dtype in info[key].items():
                        if dtype not in self._VALID_DTYPES:
                            raise ValueError(
                                f"Dtype {dtype} given for transformed column {transformed_column} is not valid. "
                                f"Acceptable dtypes: {self._VALID_DTYPES}."
                            )
                elif key == 'start_idx' or key == 'end_idx':
                    if not isinstance(info[key], int):
                        raise ValueError(f"Value for key {key} in column {column_name} is not an integer.")
                    
                elif key == "empirical_dist":
                    obj = info[key]
                    if isinstance(obj, list):
                        # Check if all elements are numbers
                        assert all(isinstance(x, (int, float)) for x in obj), "All elements in the list must be numbers."
                        array = np.array(obj)
                    elif isinstance(obj, np.ndarray):
                        # Check if it's a 1-dimensional array
                        assert obj.ndim == 1, "The numpy array must be 1-dimensional."
                        array = obj
                    else:
                        try:
                            if isinstance(obj, torch.Tensor):
                                # Check if it's a 1-dimensional tensor
                                assert obj.ndim == 1, "The tensor must be 1-dimensional."
                                array = obj.numpy()
                            else:
                                raise TypeError
                        except ImportError:
                            raise TypeError("The object must be a list, numpy array, or tensor.")
                        
                    # Check if the sum of elements is 1
                    if info['original_dtype'] in ['categorical']:
                        assert np.isclose(array.sum(), 1.0), "The sum of elements must be 1."

            # Check if start_idx, end_idx are within the valid range
            if not (0 <= info['start_idx'] <= info['end_idx'] <= info['start_idx'] + encoded_width):
                raise ValueError(
                    f"Indices start_idx {info['start_idx']} and end_idx {info['end_idx']} for column {column_name} "
                    f"are out of the valid range (0 <= start_idx <= end_idx <= {info['start_idx'] + encoded_width})."
                )

    def fit(
        self,
        data,
        batch_size=32
    ):
        """
            Train the synthesizer using the input data.

        Args:
            data: Either a pandas DataFrame (will be encoded automatically) or 
                  a torch.dataloader object containing preprocessed training data.
            batch_size (int): Batch size for DataLoader creation when input is DataFrame.
        """
        self.model.train(
            data,
            batch_size=batch_size         
        )

    def sample(self, n, condition=None, return_dataframe=False):
        """Generate synthetic samples

        Args:
            n (int): number of training samples to be generated.
            condition (torch.dataloader): dataloader contains instance level condition to be generated based on. Must have same length as n.
            return_dataframe (bool): If True, return decoded DataFrame. If False, return tensor.

        Returns:
            synth_data: Either a torch tensor (default) or pandas DataFrame containing synthesized data.
        """
        assert condition is None or len(condition) == n or len(condition) == 1, f"Condition length provided must be None, 1 or the same as number of samples! Got {len(condition)}. "

        synth_data = self.model.generate(n, condition)

        if return_dataframe and hasattr(self.model, 'decode_samples'):
            return self.model.decode_samples(synth_data)

        return synth_data

    
    def load_checkpoint(self, checkpoint):
        """ Load all model parameters and hyperparameters necessary for running a synthesizer.

        Args:
            checkpoint (dict): checkpoint items.
        """
        if checkpoint is not None:
            self.model.load_state(checkpoint)
            print("Model loaded from checkpoint!",flush=True)
        else:
            print("Model initialized!",flush=True)

    def get_checkpoint(self):
        """ Return all model parameters and hyperparameters necessary for running a synthesizer.

        Args:
            checkpoint (dict): parameter names/value pairs.
        """
        return self.model.get_state()

