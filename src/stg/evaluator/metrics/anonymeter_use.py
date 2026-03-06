from ..interfaces.metric_interface import MetricInterface
from ..interfaces.metric_interface import PlotTypes
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from anonymeter.evaluators import SinglingOutEvaluator
from anonymeter.evaluators import LinkabilityEvaluator
from anonymeter.evaluators import InferenceEvaluator

class Anonymeter(MetricInterface):
    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config=None):
        super().__init__(train_data.rename(columns=lambda x: x.replace('-', '_')),
                         holdout_data.rename(columns=lambda x: x.replace('-', '_')),
                         synth_data.rename(columns=lambda x: x.replace('-', '_')), column_name_to_datatype, config)

        if 'holdout_index' in config:
            holdout_index= config['holdout_index']
        else:
            raise Exception("holdout_index does not exit")

        control_set = holdout_data # real_data.iloc[holdout_index]
        train_set = train_data # real_data.drop(real_data.index[holdout_index])
        self.real_data=train_set.rename(columns=lambda x: x.replace('-', '_'))
        self.control_data=control_set.rename(columns=lambda x: x.replace('-', '_'))
        self.plot_type = PlotTypes.NONE  # Assuming privacy risk evaluation does not requi  re a plot by default
        self.isTable = True  # This metric provides a table of privacy risks
        self.calculate_value()
        self.calculate_table_params()

    def calculate_value(self):
        # This method will be used to execute the evaluations and store their results
        self.results = {
            'singling_out': self.evaluate_singling_out(),
            'linkability': self.evaluate_linkability(),
            'inference': self.evaluate_all_inference()
        }


    def evaluate_singling_out(self):
        print("CONTROL DATA")
        print(self.control_data)
        evaluator = SinglingOutEvaluator(ori=self.real_data, syn=self.synth_data,control= self.control_data, n_attacks=min(100, len(self.control_data)))  # Using real data as control for simplicity
        evaluator.evaluate()
        self.singling_out_evaluator=evaluator
        return {'value': evaluator.risk().value, 'ci': evaluator.risk().ci}

    def evaluate_linkability(self):
        columns = self.real_data.columns.tolist()
        if len(columns)<2:
            return  {'value': 100, 'ci': (0,0)}
        random.shuffle(columns)
        split_index = len(columns) // 2
        aux_cols = [columns[:split_index], columns[split_index:]]

        evaluator = LinkabilityEvaluator(ori=self.real_data,syn=self.synth_data, control=self.control_data,n_neighbors=1,n_attacks=50, aux_cols=aux_cols)

        evaluator.evaluate()
        self.linkability_evaluator=evaluator
        return {'value': evaluator.risk().value, 'ci': evaluator.risk().ci}

    def evaluate_all_inference(self):
        inference_results = []
        evaluators=[]
        for secret in self.real_data.columns:
            aux_cols = [col for col in self.real_data.columns if col != secret]

            evaluator = InferenceEvaluator(ori=self.real_data, syn=self.synth_data,control=self.control_data, aux_cols=aux_cols, secret=secret, n_attacks= min(50, len(self.real_data)) )

            evaluator.evaluate()
            evaluators.append(evaluator)
            inference_results.append({
                'secret': secret,
                'value': evaluator.risk().value,
                'ci': evaluator.risk().ci
            })
        self.inference_evaluators=evaluators
        return inference_results

    def calculate_table_params(self):
        self.table_params = {
            'table':pd.DataFrame( [
                {'Risk Type': 'Singling Out', 'Risk Value': self.results['singling_out']['value'], 'Confidence Interval': self.results['singling_out']['ci']},
                {'Risk Type': 'Linkability', 'Risk Value': self.results['linkability']['value'], 'Confidence Interval': self.results['linkability']['ci']}
            ] + [
                {'Risk Type': f'Inference ({res["secret"]})', 'Risk Value': res['value'], 'Confidence Interval': res['ci']}
                for res in self.results['inference']
            ])
        }

# Example usage
# real_data, synth_data should be pandas DataFrames loaded with your data
# column_name_to_datatype should be a dict mapping column names to their data types (ignored in this simplified example)
# config can be used to pass additional configurations like n_attacks, which columns to use for evaluations, etc.

# privacy_metric = PrivacyRiskMetric(real_data, synth_data, column_name_to_datatype, config)
# print(privacy_metric.value)  # Overall privacy risk

# print(privacy_metric.table_params)  # Detailed risk evaluations

