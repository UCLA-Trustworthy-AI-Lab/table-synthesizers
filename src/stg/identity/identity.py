from ..base import BaseSynthesizer
import torch

class Identity(BaseSynthesizer):
    """
        This synthesizer is for testing purpose. It saves its training data and simply return the whole training data or boostrapping from it. Note the high memory usage for large training data.
    """
    def __init__(self, data_info=None, checkpoint_interval_seconds=None, epochs=None, messageSender=None, bootstrap=False,**kwargs):
        super().__init__(data_info, checkpoint_interval_seconds, epochs, messageSender, **kwargs)
        self.bootstrap = bootstrap

    def _train(self, train_data):
        # train_data is already a DataLoader from base class
        self.train_data = train_data

    def _generate(self, n, condition=None):
        all_data = []
    
        # Collect all training data from DataLoader
        for batch in self.train_data:
            all_data.append(batch)
        all_data = torch.cat(all_data, dim=0)
        
        if self.bootstrap:
            # If bootstrap is True, sample with replacement from the observations
            indices = torch.randint(0, all_data.size(0), (n,))
            bootstrapped_data = all_data[indices]
            return bootstrapped_data
        else:
            # If bootstrap is False, return all data (legacy behavior) when n <= number of batches
            # This maintains backward compatibility with existing tests
            if hasattr(self.train_data, '__len__') and n <= len(self.train_data):
                return all_data
            else:
                # For DataFrame input or when n > number of batches, return requested samples
                if n <= all_data.size(0):
                    return all_data[:n]
                else:
                    # If more samples requested than available, cycle through the data
                    repeats = (n + all_data.size(0) - 1) // all_data.size(0)
                    repeated_data = all_data.repeat(repeats, 1)
                    return repeated_data[:n]