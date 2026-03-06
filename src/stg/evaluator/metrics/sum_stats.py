import numpy as np
import pandas as pd

from ..interfaces.metric_interface import MetricInterface, PlotTypes

class SumStats(MetricInterface):

    '''
    table_params format
    @key: Represents name of column
    @value: A dataframe representiing corresponding summary stats of the column,
            Different format for categorical vs numerical
    '''

    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config):
        super().__init__(train_data, synth_data, holdout_data, column_name_to_datatype)

        self.plot_type = PlotTypes.BOX_PLOT
        self.isTable = True  # Initialized when declaring
        self.config = config
        self.value = self.calculate_value()
        self.table_params = self.calculate_table_params()
        self.plot_params = {
            "data": self.table_params["table"],
            "skip": set([c for c in column_name_to_datatype if column_name_to_datatype[c] == 'categorical']),
        }

    
    def calculate_value(self, *args, **kwargs):
        return None

    def sum_stats(self, df_real, df_synth, metadata):
        results = {}
        try:
            columnsMeta = metadata
            datetime_cols = [c for c in columnsMeta if columnsMeta[c] == 'datetime']
            cat_cols = [c for c in columnsMeta if columnsMeta[c] == 'categorical']
            for col in datetime_cols:
                try:
                    df_real[col] = pd.to_datetime(df_real[col]).astype(int)
                    df_synth[col] = pd.to_datetime(df_synth[col]).astype(int)
                except:
                    print(f"Cannot convert column {col} to datetime! Skipping.")
                    continue
                
            for col in df_real.columns:
                try:
                    if col in cat_cols:
                        df_real_values = df_real[col].value_counts(normalize=True)
                        df_synth_values = df_synth[col].value_counts(normalize=True)
                        df_temp = pd.DataFrame(
                            data=[[df_real[col].nunique(), df_real_values.idxmax(), df_real_values.max(), (df_real_values < 0.01).sum()],
                                  [df_synth[col].nunique(), df_synth_values.idxmax(), df_synth_values.max(), (df_synth_values < 0.01).sum()]],
                            columns= ['unique_values', 'most_frequent', 'proportion_of_most_frequent', 'values_with_less_than_1%_proportion'],
                            index=['Real','Synthetic']
                        )
                        results[col] = df_temp
                    elif col in datetime_cols or np.issubdtype(df_real[col].dtype, np.number) or metadata[col] == 'numerical':
                        df_temp = pd.DataFrame(
                            data=[[df_real[col].min(), df_real[col].quantile(0.25), df_real[col].mean(), df_real[col].median(), df_real[col].quantile(0.75), df_real[col].max()],
                                  [df_synth[col].min(), df_synth[col].quantile(0.25), df_synth[col].mean(), df_synth[col].median(), df_synth[col].quantile(0.75), df_synth[col].max()]],
                            columns= ['min', '1st_quantile', 'mean', 'median', '3rd_quantile', 'max'],
                            index=['Real','Synthetic']
                        )
                        results[col] = df_temp
                except Exception as e:
                    print("Failed column " + col + " with exception: " + str(e))
                    continue
        except Exception as e:
            print(str(e))
            return None
        
        # Set plot_param using compute metrics
        self.table_params = results
    
        return {'table': results}

    '''
    Returns the table_params
    '''
    def calculate_table_params(self):
        return self.sum_stats(self.train_data, self.synth_data, self.column_name_to_datatype)
