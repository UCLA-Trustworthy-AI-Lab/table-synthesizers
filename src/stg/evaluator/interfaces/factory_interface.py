from abc import ABC, abstractmethod

class FactoryInterface(ABC):

    '''
    Factory Interface is an abstract class that can be used for MetricFactory or PlotterFactory,
    classes that provide an easier interface to initialize instances of individual metrics or plotters.
    '''

    def __init__(self, additional_instance_classes=None, *args, **kwargs):
        self.instance_classes = {}
        self.additional_instance_classes = additional_instance_classes
        if self.additional_instance_classes and type(self.additional_instance_classes) is dict:
            self.instance_classes.update(self.additional_instance_classes)
        pass

    @abstractmethod
    def create_instance(self, *args, **kwargs):
        pass