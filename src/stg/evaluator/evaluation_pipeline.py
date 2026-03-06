from .factories import *
from .evaluator import Evaluator
from .test_result_save import Saver

class EvaluationPipeline:

    '''
    Initializes and performs high-level execution of a series of metrics given a certain dataset, uses Evaluation class.
    '''

    def __init__(self, real_data, synth_data, column_name_to_datatype, config, save_path):
        self.real_data = real_data
        self.synth_data = synth_data
        self.column_name_to_datatype = column_name_to_datatype
        self.config = config
        self.save_path = save_path
        self.report = None
        self.saver = None
        self.config['meta'] = self.column_name_to_datatype

    def run_pipeline(self):
        #self.metric_factory = MetricFactory()
        #self.plotter_factory = PlotterFactory()
        #self.evaluation_factory = EvaluationFactory(self.metric_factory, self.plotter_factory, self.real_data, self.synth_data,
        #                                       self.column_name_to_datatype, self.config)
        self.evaluation_factory = None
        self.ev = Evaluator(self.evaluation_factory, real_data=self.real_data, synth_data=self.synth_data,
                       column_name_to_datatype=self.column_name_to_datatype,
                       config=self.config)
        self.ev.evaluate_all()
        self.report = self.ev.get_report()
        #print("self.save_path is:",self.save_path)
        #print(self.report)
        self.saver = Saver(reportPath=self.save_path)
        self.saver.saveReport(self.report)

    def get_report(self):
        return self.report

    def save_report(self):
        if not self.report:
            raise Exception('No report to save')
        if not self.saver:
            self.saver = Saver(reportPath=self.save_path)
            self.saver.saveReport(self.report)