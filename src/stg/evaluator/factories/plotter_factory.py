from ..plotters import *
from ..interfaces.metric_interface import PlotTypes
from ..interfaces.factory_interface import FactoryInterface


class PlotterFactory(FactoryInterface):
    def __init__(self, additional_instance_classes=None):
        super().__init__(additional_instance_classes)
        # This dictionary maps plot types to plotter classes
        self.plotter_classes = {
            PlotTypes.HEATMAP: HeatmapPlotter,
            PlotTypes.COLUMN_SHAPE: ColumnShapePlotter,
            PlotTypes.PRECISION_RECALL: PrecisionRecallPlotter,
            PlotTypes.REGRESSION_FIT: RegressionFitPlotter,
            PlotTypes.BOX_PLOT: BoxplotPlotter
        }
        self.instance_classes.update(self.plotter_classes)

    def create_instance(self, plot_type, plot_params):
        plotter_class = self.instance_classes.get(plot_type)
        if plotter_class:
            return plotter_class(plot_params)
        else:
            raise ValueError(f"No available plotter for plot type: {plot_type}")