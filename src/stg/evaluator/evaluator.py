from .factories import MetricFactory,PlotterFactory, EvaluationFactory
from sklearn.model_selection import train_test_split

class Evaluator:
    
    '''
    Executes and retrieves data for a series of metrics using EvaluationFactory class.
    '''

    def __init__(self, evaluation_factory=None, real_data=None, synth_data=None, column_name_to_datatype=None, config=None, holdout_data=None):
        print('Initialing Evaluator...')

        self.synth_data = synth_data
        self.column_name_to_datatype = column_name_to_datatype
        self.config = config

        if holdout_data is None:
            self.real_data, self.holdout_data = self.holdout_split(real_data)
        else:
            # Allows directly passing in holdout data splitted elsewhere
            self.real_data, self.holdout_data = real_data, holdout_data
        
        if evaluation_factory is None:
            metric_factory = MetricFactory()
            plotter_factory = PlotterFactory()
            evaluation_factory = EvaluationFactory(metric_factory, plotter_factory, self.real_data, self.synth_data, self.holdout_data, column_name_to_datatype, config)
        self.evaluation_factory = evaluation_factory
        self.fidelity_evaluation = None
        self.privacy_evaluation = None
        self.utility_evaluation = None

        self.create_evaluation()
        # self.evaluate_fidelity()

    def holdout_split(self, real_data):
        '''
        Perform holdout set split on real data
        '''
        # compute holdout
        has_holdout_index = "holdout_index" in self.config
        has_holdout_seed = "holdout_seed" in self.config and "holdout_size" in self.config
        assert has_holdout_index or has_holdout_seed, "[Evalution Interface Error] Holdout split must be alwasy provided for TSTR evaluation! Include holdout_index or combination of holdout seed and size."

        if has_holdout_index:
            holdout_index = self.config['holdout_index']
            holdout_index
            train_from_real, holdout_from_real = real_data.drop(holdout_index), real_data.loc[holdout_index]
        else:
            train_from_real, holdout_from_real = train_test_split(real_data, test_size=self.config['holdout_size'], random_state=self.config['holdout_seed'])
            holdout_index = holdout_from_real.index

        return train_from_real, holdout_from_real

    def create_evaluation(self):
        if self.real_data is None or self.synth_data is None or self.column_name_to_datatype is None:
            raise Exception('Must add real data, synthetic data, and column mapping before creating evaluation')
        self.fidelity_evaluation = self.evaluation_factory.create_fidelity_evaluation()
        self.privacy_evaluation = self.evaluation_factory.create_privacy_evaluation()
        self.utility_evaluation = self.evaluation_factory.create_utility_evaluation()

    def evaluate_all(self):
        for evaluation in [self.fidelity_evaluation, self.utility_evaluation, self.privacy_evaluation]:
            if evaluation is not None:
                #print(evaluation)
                evaluation.evaluate()
    
    def evaluate_fidelity(self):
        self.fidelity_evaluation.evaluate()

    def evaluate_privacy(self):
        self.privacy_evaluation.evaluate()

    def evaluate_utility(self):
        self.utility_evaluation.evaluate()

    def get_metrics(self):
        return_metrics = {}
        for evaluation in [self.fidelity_evaluation, self.utility_evaluation, self.privacy_evaluation]:
            if evaluation is not None and evaluation.is_evaluated:
                return_metrics.update(evaluation.get_metrics())
        return return_metrics
    
    def get_metric_plots(self):
        return_plots = {}
        for evaluation in [self.fidelity_evaluation, self.utility_evaluation, self.privacy_evaluation]:
            if evaluation is not None and evaluation.is_evaluated:
                return_plots.update(evaluation.get_metric_plots())

        return return_plots
    
    def get_report(self):
        return {'tables':self.get_metrics(), 'plots':self.get_metric_plots()}





