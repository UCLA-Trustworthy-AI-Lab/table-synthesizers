'''
This file contains concrete implementation of FidelityEvaluation, PrivacyEvaluation, and UtilityEvaluation
They inherit the evaluation interface
'''

from .interfaces.evaluation_interface import EvaluationInterface


class FidelityEvaluation(EvaluationInterface):

    '''
    FidelityEvaluation is a concrete implementation of EvaluationInterface
    It will calculate the fidelity metrics
    '''

    def __init__(self, metric_factory, plotter_factory, real_data, synth_data, holdout_data, column_name_to_datatype, config):
        try:
            super().__init__(metric_factory, plotter_factory, real_data, synth_data, holdout_data, column_name_to_datatype, config)
            # Load a list of metric names from config
            if 'fidelity_metrics' in self.config:
                self.metrics_to_compute = self.config['fidelity_metrics']
            else:

                self.metrics_to_compute = ["SumStats", "ColumnShape", "ColumnShapeHoldout"]# ['ColumnShape'] # ['SumStats', 'ColumnShape', 'ColumnShapeHoldout']

            print('Fidelity Evaluation Module initialized')
        except:
            print('Error initializing Fidelity Evaluation Module')

        self.metrics = {}



class PrivacyEvaluation(EvaluationInterface):

    '''
    PrivacyEvaluation is a concrete implementation of EvaluationInterface
    It will calculate the privacy metrics
    '''

    def __init__(self, metric_factory, plotter_factory, real_data, synth_data, holdout_data, column_name_to_datatype, config):
        try:
            super().__init__(metric_factory, plotter_factory, real_data, synth_data, holdout_data, column_name_to_datatype, config)
            if 'privacy_metrics' in self.config:
                self.metrics_to_compute = self.config['privacy_metrics']
            else:
                self.metrics_to_compute = ['DCR', 'Anonymeter']
            print('Privacy Evaluation Module initialized')
        except Exception as e:
            print(f'Error {e} when initializing Privacy Evaluation Module')

class UtilityEvaluation(EvaluationInterface):

    '''
    UtilityEvaluation is a concrete implementation of EvaluationInterface
    It will calculate the utility metrics
    '''

    def __init__(self, metric_factory, plotter_factory, real_data, synth_data, holdout_data, column_name_to_datatype, config):
        try:
            super().__init__(metric_factory, plotter_factory, real_data, synth_data, holdout_data, column_name_to_datatype, config)

            self.data_type = config['data_type'] if 'data_type' in config else "tabular" # ['tabular', 'timeseries']
            if 'utility_metrics' in self.config:
                self.metrics_to_compute = self.config['utility_metrics']
            else:
                self.metrics_to_compute = ["TabularUtility"]# ['TabularUtility']

            self.metadata = config['meta']
            self.target_column = config['target_columns'][0] if 'target_columns' in config else self.real_data.columns[-1]
            #self.real_data = self.preprocess_dataframe(self.real_data, config['meta'],self.target_column)
            #self.synth_data = self.preprocess_dataframe(self.synth_data, config['meta'],self.target_column)
            # self.real_data = self.preprocess(self.real_data, self.target_column)
            # self.synth_data = self.preprocess(self.synth_data, self.target_column)
            print('Utility Evaluation Module initialized')
        except:
            print('Error initializing Utility Evaluation Module')



        

