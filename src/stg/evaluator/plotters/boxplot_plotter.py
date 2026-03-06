from ..interfaces.plotter_interface import Plotter
import matplotlib.pyplot as plt
from ..interfaces.metric_interface import PlotTypes
import copy
import plotly.graph_objects as px

class BoxplotPlotter(Plotter):

    def __init__(self, plot_params):
        super().__init__(plot_type=PlotTypes.BOX_PLOT, plot_params=plot_params)

    def plot(self, verbose=False):
        print('Plotting box plots')
        data = self.plot_params["data"]
        to_skip = self.plot_params["skip"]
        
        return_tables = {}
        for c in data:
            try:
                fig = px.Figure()
                fig.update_layout(title='Boxplots for ' + c)
                if c in to_skip:
                    fig.add_trace(px.Box(y=[]))
                    fig.add_trace(px.Box(y=[]))
                else:
                    #the "data" will just be the 5 num summary
                    #drop the mean column and add it
                    
                    fig.add_trace(px.Box(y=data[c].drop("mean", axis=1).loc["Real"], name="Real")) #real
                    fig.add_trace(px.Box(y=data[c].drop("mean", axis=1).loc["Synthetic"], name="Synthetic")) #synth

                return_tables[c] = copy.deepcopy(fig) # Each fig is a plotpy object
            except Exception as e:
                print("Exception occurred: ", e, " when plotting distribution of ", c)
            if verbose:
                fig.show()

        return return_tables
