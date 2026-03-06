from abc import ABC, abstractmethod
import traceback
from ..utils.data_preprocessor import DataPreprocessor
import pandas as pd

class EvaluationInterface(ABC):

    '''
    EvaluationInterface is an abstract class that defines the interface for an evaluation object
    FidelityEvaluation, PrivacyEvaluation, and UtilityEvaluation will implement EvaluationInterface
    metrics is a list of metrics we should calculate
    '''
    def __init__(self, metric_factory, plotter_factory, real_data, synth_data, holdout_data, column_name_to_datatype, config):
        self.metric_factory = metric_factory
        self.plotter_factory = plotter_factory
        self.metrics = {}
        self.metrics_to_compute = None # a list of metrics names used to extract metrics

        self.real_data = real_data
        self.synth_data = synth_data
        self.holdout_data = holdout_data

        self.column_name_to_datatype = column_name_to_datatype
        self.config = config
        if 'meta' not in self.config:
            self.config['meta'] = self.column_name_to_datatype  # So user doens't have to do it

        self.is_evaluated = False
    
    def evaluate(self):
        '''
            Loop through all metrics and evaluate them.
        '''
        for metric in self.metrics_to_compute:
            try:
                metric_instance = self.metric_factory.create_instance(metric, self.real_data, self.synth_data, self.holdout_data, self.column_name_to_datatype, self.config)
            except Exception as e:
                print(traceback.format_exc())
                print(f"Metric {metric} calculation failed! Skipping")
                continue

            print("evaluated ", metric)
            self.metrics[metric] = metric_instance
        
        self.is_evaluated = True   

    def get_metrics(self, verbose=False):
        '''
            Get all metric tables. Each metric object is expected to have a .table_params attribute, which has a 'table' key containing a pd data frame or a dictionary of pd data frames.
        '''
        return_tables = {}
        for metric_name, metric in self.metrics.items():
            try:
                if verbose:
                    print(metric_name)
                    print(metric.table_params['table'])
                return_tables[metric_name] = metric.table_params['table']
            except Exception as e:
                print(f"Failed to get table. Skipping metrics for {metric_name}! Error encountered: {e}")
                continue
        
        return return_tables

    def get_metric_plots(self, verbose=False):
        '''
            Get all metric plots. Each metric object is expected to have a .plot_params attribute, whose format match the requirement of corresponding PlotType.
        '''
        return_plots = {}
        for metric_name, metric in self.metrics.items():
            try:
                p = self.plotter_factory.create_instance(metric.plot_type, metric.plot_params)

                return_plots[metric_name] = p.plot()
                print("Plotting ", metric_name)
                if verbose:
                    print(f"Showing plot for {metric_name}")
                    if isinstance(p, dict):
                        for col_name in p:
                            p[col_name].plot().show()
                    else:
                        p.plot().show()
            except ValueError as e:
                # Catches error from factory
                print(
                    f"No available plotter for plot type: {metric.plot_type}. Skipping plot for {metric_name}! ")
                #traceback.print_exc()
                continue
            except Exception as e:
                print(f"Failed to create plot. Skipping plot for {metric_name}! Error encountered: {e}")
                continue

        return return_plots