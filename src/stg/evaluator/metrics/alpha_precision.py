import pandas as pd

from ..interfaces.metric_interface import MetricInterface
from ..interfaces.metric_interface import PlotTypes
from ..utils.data_preprocessor import DataPreprocessor
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
import matplotlib
import numpy as np

# matplotlib.use('TkAgg')

class AlphaPrecision(MetricInterface):
    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config=None, *args, **kwargs):
        super().__init__(train_data, synth_data, holdout_data, column_name_to_datatype, config, *args, **kwargs)
        self.plot_type=None
        self.isTable = True
        self.table_params = {}
        self.data_preprocessor = DataPreprocessor()
        self.metadata = config['meta']
        self.table_params['table'] = self.calculate_table_params()

    def calculate_table_params(self):
        train_data_preprocessed, synth_data_t_preprocessed = self.data_preprocessor.transform_real_and_synth(self.train_data, self.synth_data, self.metadata)
        # print("preprocessing train and synth done")
        holdout_data_preprocessed, synth_data_h_preprocessed = self.data_preprocessor.transform_real_and_synth(self.holdout_data, self.synth_data, self.metadata)
        # print("preprocessing holdout and synth done")

        min_train_length = min(len(train_data_preprocessed), len(synth_data_t_preprocessed))
        train_data_preprocessed = train_data_preprocessed.iloc[:min_train_length]
        synth_data_t_preprocessed = synth_data_t_preprocessed.iloc[:min_train_length]

        # print("Corrected length of train data")

        min_holdout_length = min(len(holdout_data_preprocessed), len(synth_data_t_preprocessed))
        holdout_data_preprocessed = holdout_data_preprocessed.iloc[:min_holdout_length]
        synth_data_h_preprocessed = synth_data_h_preprocessed.iloc[:min_holdout_length]

        # print("Corrected length of holdout data")

        train_loader = GenericDataLoader(train_data_preprocessed)
        holdout_loader = GenericDataLoader(holdout_data_preprocessed)
        synth_t_loader = GenericDataLoader(synth_data_t_preprocessed)
        synth_h_loader = GenericDataLoader(synth_data_h_preprocessed)
        # print("Data loaded")

        ap = eval_statistical.AlphaPrecision()
        ap_train_v_synth = ap.evaluate(train_loader, synth_t_loader)
        ap_holdout_v_synth = ap.evaluate(holdout_loader, synth_h_loader)
        return {"train_vs_synth": pd.DataFrame([ap_train_v_synth]), "holdout_vs_synth": pd.DataFrame([ap_holdout_v_synth])}

    def calculate_value(self, *args, **kwargs):
        pass