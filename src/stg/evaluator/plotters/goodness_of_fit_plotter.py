from ..interfaces.plotter_interface import Plotter
import matplotlib.pyplot as plt
from ..interfaces.metric_interface import PlotTypes
import plotly.graph_objects as go
from sklearn.metrics import  r2_score
import numpy as np

def adjusted_r_squared(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

class RegressionFitPlotter(Plotter):
    def __init__(self, plot_params):
        '''
            plot_parames needs to be a dictionary with 2 keys: real and synth. Each key has a sub dictionary in format {"model_name": {"y_test":arr, "y_pred":arr, "n_features":int}} 
        '''
        super().__init__(plot_type=PlotTypes.REGRESSION_FIT, plot_params=plot_params)


    def plot(self):
        print('Plotting regression fit:')
        print(self.plot_params.keys())
        plots = {}

        # Make plots for both real/synth data
        for data_source in ['real', 'synth']:
            # Traverse all models
            for i,item in enumerate(self.plot_params[data_source].items()):
                name, params = item
                #print(params)
                y_test = params['y_test']
                y_pred = params['y_pred']
                n_features = params['n_features']
                # Calculate the fitted line using a regression model or any other fitting method
                # In this example, we'll use a simple linear regression
                fitted_line = np.polyfit(y_test, y_pred, 1)
                smoothed_y_pred = np.polyval(fitted_line, y_test)

                # Create the scatter plot
                fig = go.Figure()

                # Add the scatter plot trace
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual'))

                # Add the smoothed fitted line trace
                fig.add_trace(go.Scatter(x=y_test, y=smoothed_y_pred, mode='lines', name='Fitted Line'))

                # Update the layout
                r2_adjusted = adjusted_r_squared(y_test, y_pred, n_features)
                fig.update_layout(title=f'Scatter Plot with Fitted Line. R2 adjusted: {r2_adjusted}',
                                xaxis_title='True Values (y_test)',
                                yaxis_title='Fitted Values (y_pred)')
                
                # Save the plot for this curve.
                plots["_".join([name, data_source])] = fig
        
        return plots
        
