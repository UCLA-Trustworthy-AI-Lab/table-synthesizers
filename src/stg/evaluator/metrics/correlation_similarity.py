import numpy as np
import pandas as pd
from ..interfaces.metric_interface import MetricInterface, PlotTypes
import copy

class CorrelationSimilarity(MetricInterface):

    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config=None):
        super().__init__(train_data, synth_data, holdout_data, column_name_to_datatype, config)

        self.plot_type = PlotTypes.HEATMAP
        self.isTable = True
        self.value = self.calculate_value()
        table = self.calculate_table_params()
        self.table_params = {'table': table}
        self.plot_params = {'table': table.copy()}

    def calculate_value(self, *args, **kwargs):
        return None

    def evaluate_correlation_similarity(self, real_data, synthetic_data, numerical_attributes):
        correlation_table = pd.DataFrame(index=numerical_attributes, columns=numerical_attributes)

        for col1 in numerical_attributes:
            for col2 in numerical_attributes:
                if col1 == col2:
                    score = 1.0
                else:
                    try:
                        correlation_coefficient = real_data[[col1, col2]].corr().iloc[0, 1]
                        score = 1 - abs(correlation_coefficient)  # Transform correlation to similarity
                    except Exception as e:
                        print(f"Exception occurred with real column {col1} and synth column {col2}")
                        print("Error occurred when processing both as numerical attributes")
                        print(type(e))
                        score = np.nan

                correlation_table.at[col1, col2] = score

        return correlation_table
    
    def calculate_table_params(self, *args, **kwargs):
        numerical_attributes = [col for col, dtype in self.column_name_to_datatype.items() if dtype == 'numerical']
        s = self.evaluate_correlation_similarity(
            real_data=self.train_data,
            synthetic_data=self.synth_data,
            numerical_attributes=numerical_attributes,
        )
        return s
