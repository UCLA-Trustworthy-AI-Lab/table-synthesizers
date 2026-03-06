from abc import ABC, abstractmethod

class Plotter(ABC):
    def __init__(self, plot_type, plot_params):
        self.plot_type = plot_type
        self.plot_params = plot_params

    @abstractmethod
    def plot(self,verbose=False):
        '''
            This function should use self.plot_params to make a plot and return it. If verbose=True, the plot should be printed as well.
        '''
        pass