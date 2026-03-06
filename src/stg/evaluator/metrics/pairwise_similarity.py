import numpy as np
import pandas as pd
from sdmetrics.column_pairs import ContingencySimilarity
from sdmetrics.column_pairs import CorrelationSimilarity
from ..interfaces.metric_interface import MetricInterface, PlotTypes
import copy

class PairwiseSimilarity(MetricInterface):

    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config=None):
        super().__init__(train_data, synth_data, holdout_data, column_name_to_datatype, config)

        self.plot_type = PlotTypes.HEATMAP
        self.isTable = True
        self.value = self.calculate_value()
        table = self.calculate_table_params()
        self.table_params = {'table': table}
        self.plot_params = {'table': table.copy()}



    def evaluate_pairwise_similarity(self, real_data, synthetic_data, numerical_attributes, categorical_attributes):
        '''
        This is Xiao Yang's function with slight modifications
        Added try-catch blocks to better perceive errors
        '''

        all_attributes = numerical_attributes + categorical_attributes

        correlation_table = pd.DataFrame(index=all_attributes, columns=all_attributes)

        for col1 in all_attributes:
            for col2 in all_attributes:
                if col1 == col2:
                    score = 1.0
                elif col1 in numerical_attributes and col2 in numerical_attributes:

                    try:
                        score = CorrelationSimilarity.compute(
                            real_data=real_data[[col1, col2]],
                            synthetic_data=synthetic_data[[col1, col2]],
                            coefficient='Pearson'
                        )
                    except Exception as e:
                        print("Exception occurred with real column " + col1 + " and synth column " + col2)
                        print("Error occurred when processing both as numerical attributes")
                        print(type(e))
                elif col1 in categorical_attributes and col2 in categorical_attributes:
                    try:
                        score = ContingencySimilarity.compute(
                            real_data=real_data[[col1, col2]],
                            synthetic_data=synthetic_data[[col1, col2]]
                        )
                    except Exception as e:
                        print("Exception occurred with real column " + col1 + " and synth column " + col2)
                        print("Error occurred when processing both as categorical attributes")
                        print(type(e))
                elif col1 in numerical_attributes and col2 in categorical_attributes:
                    try:
                        score1 = self.eta_squared(real_data[col1], real_data[col2])
                        score2 = self.eta_squared(synthetic_data[col1], synthetic_data[col2])
                        score = 1 - abs(score1 - score2) / 2
                    except Exception as e:
                        print("Exception occurred with real column " + col1 + " and synth column " + col2)
                        print(
                            "Error occurred when processing first column as numerical and second column as categorical")
                        print(type(e))
                elif col2 in numerical_attributes and col1 in categorical_attributes:
                    try:
                        score1 = self.eta_squared(real_data[col2], real_data[col1])
                        score2 = self.eta_squared(synthetic_data[col2], synthetic_data[col1])
                        score = 1 - abs(score1 - score2) / 2
                    except Exception as e:
                        print("Exception occurred with real column " + col1 + " and synth column " + col2)
                        print(
                            "Error occurred when processing first column as categorical and second column as numerical")
                        print(type(e))
                else:
                    score = np.nan
                correlation_table.at[col1, col2] = score

        return correlation_table

    '''
    This is Xiao Yang's function
    '''
    def eta_squared(self, numerical_column, categorical_column):
        categories = categorical_column.unique()
        means = [numerical_column[categorical_column == category].mean() for category in categories]
        overall_mean = numerical_column.mean()

        ss_between = sum(len(numerical_column[categorical_column == category]) * (mean - overall_mean) ** 2
                         for category, mean in zip(categories, means))

        ss_total = sum((value - overall_mean) ** 2 for value in numerical_column)
        return ss_between / ss_total

    def calculate_value(self, *args, **kwargs):
        return None

    # TODO: Tackle cases where datatype is neither numerical nor categorical
    def calculate_table_params(self, *args, **kwargs):
        numerical_attributes = []
        categorical_attributes = []

        # TODO: Fix, need to decide how to treat datetime, right now we skip
        for column_name in self.column_name_to_datatype.keys():
            if self.column_name_to_datatype[column_name] == 'numerical':
                numerical_attributes.append(column_name)
            elif self.column_name_to_datatype[column_name] == 'datetime':
                continue
            else:
                categorical_attributes.append(column_name)


        s = self.evaluate_pairwise_similarity(
            real_data=self.train_data,
            synthetic_data=self.synth_data,
            numerical_attributes=numerical_attributes,
            categorical_attributes=categorical_attributes)
        return s
