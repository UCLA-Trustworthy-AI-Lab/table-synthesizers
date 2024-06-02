import torch
import dask.dataframe as dd
import numpy as np 
from base import BaseSynthesizer

DEFAULT_MODELS = {}

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
    def __init__(self, model, config, meta=None, transformer_info=None, **kwarg):
        """
        An interface that allows the construction and selection of different synthesizers. It reduces the need for changing the code each time we want to run a different synthesizer. 

        :param model: A pytorch defined model, or a name of one of the registered model here.
        :type model: BaseSynthesizer or str
        :param inferred_column_info: dictionary where keys are col names and vales are inferred column types
        """
        print(f"Initializting pytorch synthesizer. Is_cuda_enable:{torch.cuda.is_available()}")

        self.transformer_info = transformer_info
        #print("config in synthesizerWrapper",config)
        
        if isinstance(model, str) and model in DEFAULT_MODELS:
            self.model = DEFAULT_MODELS[model](**config)
        elif isinstance(model, BaseSynthesizer):
            self.model = model
        else:
            raise ValueError(f"model provided must be a BaseSynthesizer instance, or a string in the following default models: {self.get_registered_models()}. Got: {model}. ")
        self.meta = meta

    @classmethod
    def register_model(cls, new_models):
        assert isinstance(new_models, dict), "New models must be a dictionary with model name/model class pairs!"
        for md in new_models.values():
            assert issubclass(md, BaseSynthesizer), f"The model being registered must be a BaseSynthesizer instance! Received {type(md)}"
        cls._DEFAULT_MODELS.update(new_models)

    @classmethod
    def get_registered_models(cls):
        return cls._DEFAULT_MODELS

    def fit(
        self,
        data
    ):
        """
            Train the synthesizer using the preprocessed data.

        Args:
            data (torch.dataloader): a dataloader object containing preprocessed training data, in numerical format suitable for synthesizer processing. 
        """
        self.model.train(
            data            
        )

    def sample(self, n, condition=None):
        """_summary_

        Args:
            n (int): number of training samples to be generated.
            condition (torch.dataloader): dataloader contains instance level condition to be generated based on. Must have same length as n.

        Returns:
            synth_data: a torch tensor containing synthesized data in numerical format. The preprocessor will convert it back to table format.
        """
        assert len(condition) == n or len(condition) == 1, f"Condition provided must be 1 or the same as number of samples! Got {len(condition)}. "

        synth_data = self.model.generate(n, condition)

        #print("type(synth_data) in wrapper:",type(synth_data))

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

