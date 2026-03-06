from ..interfaces.plotter_interface import Plotter
from ..interfaces.metric_interface import PlotTypes
import plotly.graph_objects as go


class HeatmapPlotter(Plotter):
    def __init__(self, plot_params):
        super().__init__(plot_type=PlotTypes.HEATMAP, plot_params=plot_params)

    def plot(self):
        plot_matrix = self.plot_params['table']
        # Assuming plot_params are in the format required by Plotly's heatmap, i.e., a 2D array of numbers
        fig = go.Figure(data=go.Heatmap(
            z=plot_matrix.astype(float), # Ensure all values are float
            colorscale='RdBu', # This is a diverging color scale similar to 'coolwarm'
            zmin=0,  # Minimum color value
            zmax=1   # Maximum color value
        ))

        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_nticks=36,
            yaxis_nticks=36
        )
        # fig.show()  # This will render the plotly figure in a web browser or inline if using a Jupyter notebook
        return fig