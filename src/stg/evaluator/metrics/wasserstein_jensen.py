from ..interfaces.metric_interface import MetricInterface
from ..interfaces.metric_interface import PlotTypes
from ..utils.data_preprocessor import DataPreprocessor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import time


class WassersteinJensen(MetricInterface):

    '''
    Computes a matrix of differences between train and synthetic, and holdout and synthetic.
    Wasserstein distance for numeric, and Jensen distance for categorical.
    '''

    def __init__(self, train_data, holdout_data, synth_data, column_name_to_datatype, config=None):

        super().__init__(train_data, holdout_data, synth_data, column_name_to_datatype, config)


        self.plot_type=None

        # in order to create a result csv, set self.table_params['table'] to a dataframe. This will create <Classname>.csv
        # to create multiple csv's, set self.table_params['table']['name'] to different dataframes
        # then this will make <Classname>_name.csv

        # self.continuous = [c for c in column_name_to_datatype if column_name_to_datatype[c] == 'continuous']
        # self.categorical = [c for c in column_name_to_datatype if column_name_to_datatype[c] == 'categorical']
        self.table_params = {'table': {'real_v_synth': None, 'holdout_v_synth': None}}
        self.table_params['table']['real_v_synth'] = self.calculate_table_params(train_data, synth_data)
        self.table_params['table']['holdout_v_synth'] = self.calculate_table_params(holdout_data, synth_data)

    def calculate_table_params(self, df1, df2):

        df = pd.DataFrame(columns = ["column", "metric", "score"])

        for col in self.column_name_to_datatype:

            
            if self.column_name_to_datatype[col] == 'numerical':
                
                L_col = df1[col].fillna(df1[col].mean())
                R_col = df2[col].fillna(df2[col].mean())
                
                # Replace append with concat
                new_row = pd.DataFrame({"column": [col], "metric": ["wasserstein"], "score": [wasserstein_distance(L_col, R_col)]})
                df = pd.concat([df, new_row], ignore_index=True)
                
            if self.column_name_to_datatype[col] == 'categorical':
                
                L_col = df1[col].fillna(df1[col].mode()[0])# .astype('category').cat.codes
                R_col = df2[col].fillna(df2[col].mode()[0])# .astype('category').cat.codes  

                freq1 = df1[col].value_counts(normalize=True)
                freq2 = df2[col].value_counts(normalize=True)          
    
                if df2[col].dtype == 'object':
                    L_col = self.string_conv_int(L_col)
                    R_col = self.string_conv_int(R_col)

                # Create a DataFrame to align and fill missing categories with 0
                freq_df = pd.DataFrame({'freq1': freq1, 'freq2': freq2}).fillna(0)

                # Compute the Jensen-Shannon divergence
                js_divergence = jensenshannon(freq_df['freq1'], freq_df['freq2'])

                # Replace append with concat
                new_row = pd.DataFrame({"column": [col], "metric": ["jensenshannon"], "score": [js_divergence]})
                df = pd.concat([df, new_row], ignore_index=True)
        
        return df


    def string_conv_int(self, x):
        mapping = {v: i for i, v in enumerate(set(x))}
        return np.array(list(map(mapping.__getitem__, x)))

    def calculate_value(self, *args, **kwargs):
        pass

