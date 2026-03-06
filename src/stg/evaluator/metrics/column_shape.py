from ..interfaces.metric_interface import MetricInterface
from ..interfaces.metric_interface import PlotTypes

import pandas as pd
from sdmetrics.reports.single_table import QualityReport

class ColumnShape(MetricInterface):
    '''
        This function calls the sdmetrics quality report and returns the shape similarity of each column.
    '''

    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config=None):

        # self was removed as the first arg here for py3 syntax - minh
        super().__init__(train_data, synth_data, holdout_data, column_name_to_datatype, config)

        self.value = None

        # BEGIN CHANGE
        # self.table_params = self.calculate_table_params()

        train = self.train_data.reset_index().drop(columns=['index'])
        holdout = self.holdout_data.reset_index().drop(columns=['index'])
        synth = self.synth_data

        self.table_params = {'table': {'real': None, 'holdout': None}}
        trainvsynth = self.calculate_table_params(left = train, right = synth)
        self.table_params['table']['real'] = trainvsynth['table']

        holdoutvsynth = self.calculate_table_params(left = holdout, right = synth)
        self.table_params['table']['holdout'] = holdoutvsynth['table']

        self.plot_type = PlotTypes.COLUMN_SHAPE
        self.plot_params = {"real_table": train_data.reset_index().drop(columns=['index']),"synthetic_table":synth_data, "metadata":self.convert_to_sdv()}
        self.isTable = True # A table where each row shows the score of one column



    def convert_to_sdv(self):
        STANDARD_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
        # Convert to sdv format
        metadata = {'primary_key':None, 'columns':{}}

        meta = self.column_name_to_datatype
        for c in meta:

          # Update 04052023: specify more categories based on meta created by inferring real data.
          # Update 05042923: change metadata format.      
          # Update 07122023: add support for "binary" type as categorical
          if meta[c] == 'categorical' or meta[c] == 'binary':
              metadata['columns'][c] = {'sdtype': "categorical"}
          elif meta[c] == 'ordinal':
              metadata['columns'][c] = {"sdtype": "numerical","subtype": "integer"}
          elif meta[c] == 'continuous' or meta[c] == 'numerical':   # Note check this
              metadata['columns'][c] = {"sdtype": "numerical","subtype": "float"}
          elif meta[c] == 'datetime':
              # Convert to int for eval.
              #data[c] = (pd.to_datetime(data[c]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') // 10**9
              #sampled[c] = (pd.to_datetime(sampled[c]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') // 10**9

              self.real_data[c] = pd.to_datetime(self.real_data[c], errors='coerce')
              self.synth_data[c] = pd.to_datetime(self.synth_data[c], errors='coerce')
              metadata['columns'][c] = {"sdtype": "datetime", "datetime_format": STANDARD_DATETIME_FORMAT}

        return metadata

    def calculate_table_params(self, left, right, *args, **kwargs):
        report = QualityReport()
        
        try:

            # param assignment
            # holdout_data -> self.holdout_data (seems like will also need reset index, drop index)
            # column shape is a binary comparison.
            # so rn column_shape_holdout is doing holdout vs synth, column_shape is doing train vs synth. COMBINE

            # Only evaluate on the train data
            meta = self.convert_to_sdv()
            #train_data = self.real_data
            report.generate(left, right, meta)

            column_shapes_info_df = report.get_details(property_name='Column Shapes')
            df_avg = pd.DataFrame({"Column":"All", "Metric":"Average Shape", "Score":column_shapes_info_df['Score'].mean()},index=[column_shapes_info_df.shape[0]])

            column_correlation_info_df = report.get_details(property_name='Column Pair Trends')
            column_correlation_info_df['Column'] = column_correlation_info_df['Column 1'] + " and " + column_correlation_info_df['Column 2']
            column_correlation_info_df = column_correlation_info_df[['Column', "Metric", "Score"]]
            df_avg_corr = pd.DataFrame({"Column":"All Pairs", "Metric":"Average Corr Similarity", "Score":column_correlation_info_df['Score'].mean()},index=[column_shapes_info_df.shape[0]])
            return {'report': report, 'table': pd.concat([column_shapes_info_df, df_avg, column_correlation_info_df, df_avg_corr])}
        except Exception as e:
            print("Exception occurred: ", e, " when calculating colum quality", flush=True)
    
    def calculate_value(self, real_data, synth_data, *args, **kwargs):
        # Implement the calculation logic, didnt need this one
        return None
