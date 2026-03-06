from ..interfaces.metric_interface import MetricInterface

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.spatial.distance import cdist
import scipy

import dask.dataframe as dd
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def distance_closest_record(train_df, holdout_df, syn_df, npartitions=5, holdout_index=[]):
    

    """
        Function based on: https://github.com/UCLA-Trustworthy-AI-Lab/Autodiff/blob/main/src/models/autodiff/Evaluation.py
    """
    
    # Function to encode string columns as integers
    def encode_integer(df):
        string_columns = df.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for col in string_columns:
            df[col] = label_encoder.fit_transform(df[col])
        return df

    # Preprocess DataFrames
    real_df = train_df.fillna(0)
    syn_df = syn_df.fillna(0)
    holdout_df = holdout_df.fillna(0)

    # Combine real, holdout, and synthetic data
    num_real_data = real_df.shape[0]
    num_holdout_data = holdout_df.shape[0]
    combined_df = pd.concat([real_df, holdout_df, syn_df])
    combined_df = encode_integer(combined_df)

    # Correctly slice the combined dataframe
    real_df = combined_df[:num_real_data]
    holdout_df = combined_df[num_real_data:num_real_data + num_holdout_data]
    syn_df = combined_df[num_real_data + num_holdout_data:]

    

    # Split holdout df
    # Split here to align train/holdout dimensions
    # real_df,holdout_df = real_df.drop(holdout_index), real_df.loc[holdout_index]

    # Convert to Dask DataFrames
    real_ddf = dd.from_pandas(real_df, npartitions=npartitions)
    syn_ddf = dd.from_pandas(syn_df, npartitions=npartitions)
    holdout_ddf = dd.from_pandas(holdout_df, npartitions=npartitions)

    # Function to compute the L2 distance between two DataFrames
    def l2_distance_matrix(df1, df2):
        # Convert to numpy arrays first to avoid Dask recursion
        df1_array = df1.values if isinstance(df1, pd.DataFrame) else df1
        df2_array = df2.values if isinstance(df2, pd.DataFrame) else df2
        return cdist(df1_array, df2_array, metric='euclidean')

    # Function to compute the minimum L2 distance for each row in syn_df with respect to another DataFrame
    def compute_min_l2_distance(row, df):
        # Convert df to numpy array if it's not already
        df_array = df.values if isinstance(df, pd.DataFrame) else df
        row_array = row.values if isinstance(row, pd.Series) else row
        distances = cdist(row_array.reshape(1, -1), df_array, metric='euclidean')
        return np.min(distances)

    # Convert Dask DataFrames to pandas before computing distances
    real_df_computed = real_ddf.compute()
    syn_df_computed = syn_ddf.compute()
    holdout_df_computed = holdout_ddf.compute()

    # Calculate distances using pandas DataFrames
    Min_L2_Distance_Real = syn_df_computed.apply(
        lambda x: compute_min_l2_distance(x, real_df_computed), 
        axis=1
    )
    Min_L2_Distance_Holdout = syn_df_computed.apply(
        lambda x: compute_min_l2_distance(x, holdout_df_computed), 
        axis=1
    )

    # Create the results DataFrame
    syn_df_result = syn_df_computed.copy()
    syn_df_result['Min_L2_Distance_Real'] = Min_L2_Distance_Real
    syn_df_result['Min_L2_Distance_Holdout'] = Min_L2_Distance_Holdout

    # Calculate the proportion of points closer to real_df than holdout_df
    # Ensure we're working with computed values
    Min_L2_Distance_Real = syn_df_result['Min_L2_Distance_Real']
    Min_L2_Distance_Holdout = syn_df_result['Min_L2_Distance_Holdout']
    
    closer_to_real_count = (Min_L2_Distance_Real < Min_L2_Distance_Holdout).sum()
    proportion_closer_to_real = closer_to_real_count / len(syn_df_result)

    # Calculate column means using the computed values
    mean_distances = {
        'Min_L2_Distance_Real': float(Min_L2_Distance_Real.mean()),
        'Min_L2_Distance_Holdout': float(Min_L2_Distance_Holdout.mean())
    }

    # Create a one-row DataFrame with these means
    mean_df = pd.DataFrame([mean_distances])
    mean_df['Proportion Closer to Real'] = proportion_closer_to_real

    return mean_df

class DCR(MetricInterface):
    '''
    @param1 real_data: the real data in Pandas Dataframe
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
        self.isTable = True         # Initialized when declaring

        self.train_data = train_data
        self.holdout_data = holdout_data

        self.synth_data = synth_data
        self.column_name_to_datatype = column_name_to_datatype
        self.config = config
        self.plot_params = None        #
        self.npartitions = config['npartitions_dcr'] if "npartitions_dcr" in config else 5

        # Confirm holdout_index is provided
        # Note: TSTR methods requires a holdout/test data that are never included for synthesizer training. 
        # The utility testing must follow the same splitting provided

        # assert "holdout_index" in config, "Index of holdout set must be provided for TSTR evaluation!"
        # self.holdout_index = config['holdout_index']

        self.table_params = self.calculate_table_params(self.train_data, self.holdout_data, self.synth_data,self.npartitions)

    '''
    Returns the value of the metric
    '''
    def calculate_value(self, *args, **kwargs):
        pass

    '''
    Returns the table_params
    '''

    def calculate_table_params(self, train, holdout, synth,npartitions):
        return {'table':distance_closest_record(train, holdout, synth,npartitions)}

