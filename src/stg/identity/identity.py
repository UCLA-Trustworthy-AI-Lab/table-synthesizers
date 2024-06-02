from ..base import BaseSynthesizer
import torch

class Identity(BaseSynthesizer):
    """
        This synthesizer is for testing purpose. It saves its training data and simply return the whole training data or boostrapping from it. Note the high memory usage for large training data.
    """
    def __init__(self, data_info, checkpoint_interval_seconds=None, epochs=None, messageSender=None, bootstrap=False,**kwargs):
        super().__init__(data_info, checkpoint_interval_seconds, epochs, messageSender, **kwargs)
        self.bootstrap = bootstrap

    def _train(self, train_data):
        self.train_data = train_data

    def _generate(self, n, condition=None):
        all_data = []
    
        if self.bootstrap:
            # If bootstrap is True, sample with replacement from the observations
            for batch in self.train_data:
                all_data.append(batch)
            all_data = torch.cat(all_data, dim=0)
            indices = torch.randint(0, all_data.size(0), (n,))
            bootstrapped_data = all_data[indices]
            return bootstrapped_data
        else:
            # If bootstrap is False, simply concatenate all batches
            for batch in self.train_data:
                all_data.append(batch)
            concatenated_data = torch.cat(all_data, dim=0)
            return concatenated_data