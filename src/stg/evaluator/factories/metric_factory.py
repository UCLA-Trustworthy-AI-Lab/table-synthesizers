from ..metrics import *
from ..interfaces.factory_interface import FactoryInterface
import os

class MetricFactory(FactoryInterface):

    '''
    Provides simple interface to initialize Metric instances based on keywords and passes down data.
    '''

    def __init__(self, additional_instance_classes=None):
        # This dictionary maps metric identifiers to metric classes
        super().__init__(additional_instance_classes)
        self.metric_classes = {
            'ColumnShape': ColumnShape,
            'CorrelationSimilarity': CorrelationSimilarity,
            'DCR': DCR,
            'PairwiseSimilarity': PairwiseSimilarity,
            'SumStats': SumStats,
            'TabularUtility': TabularUtility,
            "WassersteinJensen": WassersteinJensen, 
            "AllColumnCorrelations": AllColumnCorrelations,
            "SynthMIA": SynthMIA,
        }
        if 'ALPHA_PRECISION_AVAILABLE' in globals() and ALPHA_PRECISION_AVAILABLE and AlphaPrecision is not None:
            self.metric_classes["AlphaPrecision"] = AlphaPrecision
        self.instance_classes.update(self.metric_classes)

    def create_instance(self, metric_id, real_data, synth_data, holdout_data, column_name_to_datatype, config):
        metric_class = self.instance_classes.get(metric_id)
        if metric_class:
            return metric_class(real_data, synth_data, holdout_data, column_name_to_datatype, config)
        else:
            raise ValueError(f"Cannot get metric {metric_id}!")
