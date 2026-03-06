from abc import ABC, abstractmethod
from enum import Enum

# Acceptable PlotTypes
class PlotTypes(Enum):
    HEATMAP = "HEATMAP_IN_GRID"
    COLUMN_SHAPE = "DISTRIBUTION_OF_AND_REAL_SYNTHETIC_COLUMN"
    PRECISION_RECALL = "PRECISION_RECALL_CURVE_FOR_CLASSIFICATION"
    REGRESSION_FIT = "REGRESSION_GOODNESS_OF_FIT"
    BOX_PLOT = "BOX_PLOT"
    NONE = "NO_PLOT_FOR_THIS_METRIC"

# Base Metric Interface
'''
MetricInterface is an abstract class that defines the interface for a metric object
Note: consider adding more functions that aid plotting and tabling of data
@param1 real_data: the real data, passed in by reference from the relevant evaluation object
@param2 synth_data: the synthetic data, passed in by reference from the relevant evaluation object
'''
class MetricInterface(ABC):
    '''
    @param1 real_data: the real data in Pandas Dataframe used to train synthesizer
    @param2 synth_data: the synthetic data in Pandas Dataframe
    @param3 column_name_to_datatype: a dictionary mapping column names to their data types, can be 'numerical', 'categorical', or 'datetime'
    @param4 config: a dictionary containing any other configurations, such as target variable for classification, and number of tapas attack rounds.

    @field1 value: the value of the metric, for metrics with a single number as value
    @field2 table_params: a dictionary representing the metric as table
    @field3 plot_type: a string representing the type of plot to use for the metric
    @field4 isTable: a boolean representing whether the metric is a table or a single value
    @field5 plot_params: a dictionary containing data used for plotting. 
    '''

    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config=None, *args, **kwargs):

        self.value = None
        self.table_params = {'table':None}
        self.plot_type = None
        self.isTable = None         # Initialized when declaring

        self.train_data = train_data
        self.synth_data = synth_data
        self.holdout_data = holdout_data

        self.column_name_to_datatype = column_name_to_datatype


        self.config = config
        self.plot_params = None        #

    '''
    Returns the value of the metric
    '''
    @abstractmethod
    def calculate_value(self, *args, **kwargs):
        pass

    '''
    Returns the table_params
    '''
    @abstractmethod
    def calculate_table_params(self):
        pass
