from ..interfaces.plotter_interface import Plotter
from ..interfaces.metric_interface import PlotTypes
import copy

try:
    from sdmetrics.reports.utils import get_column_plot
except ImportError:
    from sdmetrics.visualization import get_column_plot

class ColumnShapePlotter(Plotter):

    '''
    plot_params is a report object from sdmetrics
    '''
    def __init__(self, plot_params):
        super().__init__(plot_type=PlotTypes.COLUMN_SHAPE, plot_params=plot_params)

    def plot(self, verbose=False):
        print('Plotting column shape')
        real_table = self.plot_params['real_table']
        synthetic_table = self.plot_params['synthetic_table']
        metadata = self.plot_params['metadata']
        columns = real_table.columns

        return_tables = {}

        for c in columns:
            try:
                fig = get_column_plot(
                    real_data=real_table,
                    synthetic_data=synthetic_table,
                    metadata=metadata,
                    column_name=c,
                )
                return_tables[c] = copy.deepcopy(fig) # Each fig is a plotpy object
            except Exception as e:
                print("Exception occurred: ", e, " when plotting distribution of ", c)
            if verbose:
                fig.show()

        return return_tables