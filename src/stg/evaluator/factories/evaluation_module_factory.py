from ..evaluation_modules import FidelityEvaluation, PrivacyEvaluation, UtilityEvaluation
from .metric_factory import MetricFactory
from .plotter_factory import PlotterFactory
import pandas as pd

class EvaluationFactory:

    # TODO: Check datatype of dataframes

    def __init__(
            self, 
            metric_factory: MetricFactory, 
            plotter_factory: PlotterFactory, 
            real_data: pd.DataFrame, 
            synth_data: pd.DataFrame, 
            holdout_data: pd.DataFrame, 
            column_name_to_datatype: dict[str, str], 
            config: dict[object, object]
        ):
        """
        Initializes Evaluation Module
        """
        self.real_data = real_data
        self.synth_data = synth_data
        self.holdout_data = holdout_data
        self.column_name_to_datatype = column_name_to_datatype
        self.config = config
        self.metric_factory = metric_factory
        self.plotter_factory = plotter_factory

    def create_fidelity_evaluation(self) -> FidelityEvaluation:
        """
        Creates an instance of the FidelityEvaluation class
        """
        return FidelityEvaluation(self.metric_factory, self.plotter_factory, self.real_data, self.synth_data, self.holdout_data, self.column_name_to_datatype, self.config)

    def create_privacy_evaluation(self) -> PrivacyEvaluation:
        """
        Creates an instance of the PrivacyEvaluation class
        """
        return PrivacyEvaluation(self.metric_factory, self.plotter_factory, self.real_data, self.synth_data, self.holdout_data, self.column_name_to_datatype, self.config)

    def create_utility_evaluation(self) -> UtilityEvaluation:
        """
        Creates an instance of the UtilityEvaluation class
        """
        return UtilityEvaluation(self.metric_factory, self.plotter_factory, self.real_data, self.synth_data, self.holdout_data, self.column_name_to_datatype, self.config)