from ..interfaces.plotter_interface import Plotter
from ..interfaces.metric_interface import PlotTypes
import plotly.graph_objects as go
from sklearn.metrics import  r2_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, accuracy_score

class PrecisionRecallPlotter(Plotter):
    def __init__(self, plot_params):
        '''
            plot_parames needs to be a dictionary with 2 keys: real and synth. Each key has a sub dictionary in format {"model_name": {"y_test_bin":arr, "y_score":arr}} 
        '''
        super().__init__(plot_type=PlotTypes.PRECISION_RECALL, plot_params=plot_params)


    def plot(self):
        plots = {}
        for data_source, models_params in self.plot_params.items():
            for model_name, params in models_params.items():
                y_test_bin, y_score = params['y_test_bin'], params['y_score']

                # Flatten y_test_bin and y_score for micro-average calculation
                y_test_flat = y_test_bin.ravel()
                if y_score.shape[1] == 2:  # Check if binary classification
                    y_score_flat = y_score[:, 1]  # Use scores for the positive class
                else:
                    y_score_flat = y_score.ravel()


                precision, recall, _ = precision_recall_curve(y_test_flat, y_score_flat)

                # Exclude the last point
                precision = precision[:-1]
                recall = recall[:-1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Micro-average'))

                fig.update_layout(
                    title=f'Micro-Average Precision-Recall Curve for {model_name} ({data_source})',
                    xaxis=dict(title='Recall', range=[0, 1]),
                    yaxis=dict(title='Precision', range=[0, 1]),
                    autosize=False,
                    width=600,
                    height=600
                )

                plot_key = f"{model_name}_{data_source}"
                plots[plot_key] = fig

        return plots
        
