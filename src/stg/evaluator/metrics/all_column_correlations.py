from ..interfaces.metric_interface import MetricInterface

import pandas as pd
import numpy as np

from dython.nominal import theils_u


class AllColumnCorrelations(MetricInterface):

    '''
    Computes a matrix of distances between train and synthetic, and holdout and synthetic.
    Uses Theils-U correlation and correlation rtio.
    '''

    def __init__(self, train_data, holdout_data, synth_data, column_name_to_datatype, config=None):

        super().__init__(train_data, holdout_data, synth_data, column_name_to_datatype, config)


        self.plot_type=None

        # in order to create a result csv, set self.table_params['table'] to a dataframe. This will create <Classname>.csv
        # to create multiple csv's, set self.table_params['table']['name'] to different dataframes
        # then this will make <Classname>_name.csv

        # self.continuous = [c for c in column_name_to_datatype if column_name_to_datatype[c] == 'continuous']
        # self.categorical = [c for c in column_name_to_datatype if column_name_to_datatype[c] == 'categorical']
        # self.table_params = {'table': {'real_v_synth': None, 'holdout_v_synth': None}}
        # self.table_params['table']['real_v_synth'] = self.calculate_table_params(train_data, synth_data)
        # self.table_params['table']['holdout_v_synth'] = self.calculate_table_params(holdout_data, synth_data)

        # format of output tables will be:
        # Category 1, Category 2, Metric Type, Result

        

        self.table_params['table'] = self.calculate_table_params()

    def theils_u_mat(self, df):
        # Compute Theil's U-statistics between each pair of columns
        cate_columns = df.shape[1]
        theils_u_mat = np.zeros((cate_columns, cate_columns))

        for i in range(cate_columns):
            for j in range(cate_columns):
                theils_u_mat[i, j] = theils_u(df.iloc[:, i], df.iloc[:, j])

        return theils_u_mat

    def correlation_ratio(self, categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        
        for i in range(cat_num):
            # Use .iloc to select by integer position
            cat_measures = measurements.iloc[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
        denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
        
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator / denominator
        
        return eta

    def ratio_mat(self, df, continuous_columns, categorical_columns):
        rat_mat = pd.DataFrame(index=continuous_columns, columns=categorical_columns)
        
        if len(categorical_columns) == 0 or len(continuous_columns) == 0:
            return np.zeros(1)
        else:
            for cat_col in categorical_columns:
                for cont_col in continuous_columns:
                    rat_mat[cat_col][cont_col] = self.correlation_ratio(df[cat_col], df[cont_col])
        return rat_mat.values

    def fillNa_cont(self, df):
        for col in df.columns:
            mean_values = df[col].mean()
            df[col].fillna(mean_values, inplace=True)
        return df

    def fillNa_cate(self, df):
        for col in df.columns:
            mode_values = df[col].mode()[0]
            df[col].fillna(mode_values, inplace=True)
        return df

    def compute_correlation(self, df, continuous_columns, categorical_columns):

        num_mat = pd.DataFrame(df[continuous_columns])
        cat_mat = pd.DataFrame(df[categorical_columns])
        
        num_mat = self.fillNa_cont(num_mat)
        cat_mat = self.fillNa_cate(cat_mat)
        
        pearson_sub_matrix = np.corrcoef(num_mat, rowvar = False)
        theils_u_matrix = self.theils_u_mat(cat_mat)
        correl_ratio_mat = self.ratio_mat(df, continuous_columns, categorical_columns)
    
        return (pearson_sub_matrix, theils_u_matrix, correl_ratio_mat)

    def calculate_table_params(self):
        print(self.synth_data)
        print(self.train_data)
        categorical_columns = [key for key in self.column_name_to_datatype if self.column_name_to_datatype[key] == 'categorical']
        continuous_columns = [key for key in self.column_name_to_datatype if self.column_name_to_datatype[key] == 'numerical']

        syn_pearson, syn_theils, syn_ratio = self.compute_correlation(self.synth_data, continuous_columns, categorical_columns)
        train_pearson, train_theils, train_ratio = self.compute_correlation(self.train_data, continuous_columns, categorical_columns)
        holdout_pearson, holdout_theils, holdout_ratio = self.compute_correlation(self.holdout_data, continuous_columns, categorical_columns)

       
        # Calculate differences
        train_synthetic_pearson_diff = np.linalg.norm(train_pearson - syn_pearson)
        train_synthetic_theils_diff = np.linalg.norm(train_theils - syn_theils)
        train_synthetic_ratio_diff = np.linalg.norm(train_ratio - syn_ratio)

        holdout_synthetic_pearson_diff = np.linalg.norm(holdout_pearson - syn_pearson) 
        holdout_synthetic_theils_diff = np.linalg.norm(holdout_theils - syn_theils)
        holdout_synthetic_ratio_diff = np.linalg.norm(holdout_ratio - syn_ratio)

        # Create the rows
        data = [
            ['Train', 'Synthetic', train_synthetic_pearson_diff, train_synthetic_theils_diff, train_synthetic_ratio_diff],
            ['Holdout', 'Synthetic', holdout_synthetic_pearson_diff, holdout_synthetic_theils_diff, holdout_synthetic_ratio_diff],
        ]

        # Create the DataFrame
        df = pd.DataFrame(data, columns=['Dataset 1', 'Dataset 2', 'Pearson difference', 'Theils Difference', 'Correlation Ratio Difference'])

        return df


    def string_conv_int(self, x):
        mapping = {v: i for i, v in enumerate(set(x))}
        return np.array(list(map(mapping.__getitem__, x)))

    def calculate_value(self, *args, **kwargs):
        pass

