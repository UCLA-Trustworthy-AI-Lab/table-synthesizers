from ..interfaces.metric_interface import MetricInterface
from ..interfaces.metric_interface import PlotTypes
from ..utils.data_preprocessor import DataPreprocessor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

from catboost import CatBoostClassifier

import pandas as pd
import time


class TabularUtility(MetricInterface):
    '''
        This function fits machine learning models on downstream tasks using real/synthetic data and then compute performance metrics on the real test set.
    '''

    def __init__(self, train_data, synth_data, holdout_data,  column_name_to_datatype, config=None, *args, **kwargs):
        super().__init__(train_data, synth_data,holdout_data,   column_name_to_datatype, config)
        

        self.train_data = train_data
        self.synth_data = synth_data
        self.holdout_data = holdout_data
        self.value = None
        self.table_params = {}
        self.isTable = True # A table showing the score of each model, each row is a model

        # Currently only accept one target column
        self.metadata = column_name_to_datatype

        self.target_column = config['target_column'] # if 'target_column' in config else train_data.columns[-1]

        self.task = self.determine_task() if "target_task" not in config else config['target_task']

        print("Target column is: " + self.target_column)

        # Get target column from config. Used the last column if not specified.
        # Then infer the utility task based on column type.
        # Finally change plot type accordingly.
        if self.task == 'regression':
            self.plot_type = PlotTypes.REGRESSION_FIT
        else:
            self.plot_type = PlotTypes.PRECISION_RECALL


        # unpacking values to set table_params
        self.table_params['table'], self.plot_params = self.calculate_table_params()

    def determine_task(self):
        '''
            Determine plot type and task type based on the target data type given.
        '''
        target_column_type = self.metadata[self.target_column]

        if target_column_type in ['numerical', 'continuous', 'ordinal', 'datetime']:
            return 'regression'

        elif target_column_type == 'binary' or len(pd.unique(self.train_data[self.target_column])) == 2:
            return 'binary_classification'
        else:
            return 'multi_classification'

    def preprocess(self, source):
        test_df = self.holdout_data
        if source == 'real':
            train_df = self.train_data
        elif source == 'synth':
            train_df = self.synth_data
        elif source == "augmented":
            train_df = pd.concat([self.train_data, self.synth_data])

        target_column = self.target_column

        X_train, y_train = train_df.drop(target_column, axis=1), train_df[[target_column]]
        X_test, y_test = test_df.drop(target_column, axis=1), test_df[[target_column]]
        preprocessor = DataPreprocessor(target_column)

        X_train = preprocessor.fit_transform_train(X_train,self.metadata)
        y_train = preprocessor.fit_transform_train(y_train,self.metadata).squeeze()
        X_test = preprocessor.transform_test(X_test,self.metadata)
        y_test = preprocessor.transform_test(y_test,self.metadata).squeeze()

        return X_train, y_train, X_test, y_test

    def calculate_table_params(self):
        data_sources = ['real', "synth", "augmented"]
        metric_params, plots_params = {}, {}

        for source in data_sources:
            X_train, y_train, X_test, y_test = self.preprocess(source)
        
            if self.task == 'regression':
                metric_params[source], plots_params[source] = self.regression_one_column(X_train, X_test, y_train, y_test)
            else:
                metric_params[source], plots_params[source] = self.run_classification_one_column(X_train, X_test, y_train, y_test)

            print(f"Fitted {source} data!")
    
        return metric_params, plots_params
       
    def adjusted_r_squared(self, y_true, y_pred, n_features):
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    def regression_one_column(self, X_train, X_test, y_train, y_test):
    
        # Define regression models
        models = {
            'LinearRegression': LinearRegression(),
            'Lasso': Lasso(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(),
            #'MLPRegressor': MLPRegressor(max_iter=500)
        }

        # DataFrame to store metrics
        metrics = pd.DataFrame(columns=['Model', 'Test MSE', 'Test R2', 'Test Adjusted R2', 'Test MAE', 'Test RMSE', 'Train MSE', 'Train R2', 'Train Adjusted R2', 'Train MAE', 'Train RMSE'])
        plot_params = {}

        # Apply each model
        for i, item in enumerate(models.items()):
            name, model = item
            # Fit the model
            print(f"Fitting model {name}")
            model.fit(X_train, y_train)

            # Make predictions on test set
            y_test_pred = model.predict(X_test)
            # Calculate test metrics
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_adjusted_r2 = self.adjusted_r_squared(y_test, y_test_pred, X_test.shape[1])
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)

            # Make predictions on train set
            y_train_pred = model.predict(X_train)
            # Calculate train metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            train_adjusted_r2 = self.adjusted_r_squared(y_train, y_train_pred, X_train.shape[1])
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)

            # Append results to the metrics dataframe
            metrics = pd.concat([metrics, pd.DataFrame({
                'Model': name, 
                'Test MSE': test_mse, 
                'Test R2': test_r2, 
                'Test Adjusted R2': test_adjusted_r2, 
                'Test MAE': test_mae, 
                'Test RMSE': test_rmse, 
                'Train MSE': train_mse, 
                'Train R2': train_r2, 
                'Train Adjusted R2': train_adjusted_r2, 
                'Train MAE': train_mae, 
                'Train RMSE': train_rmse
            }, index=[i])])
            plot_params[name] = {"y_test": y_test, "y_pred": y_test_pred, "n_features": X_train.shape[1]}

        return metrics, plot_params
    
            
    def calculate(self, real_data, synth_data, *args, **kwargs):
        # Call corresponding method for based on inferred target column type.
        pass

    def run_classification_one_column(self, X_train, X_test, y_train, y_test):

        target_column = self.target_column

        # Define classification models with reduced complexity for testing
        models = {
            'NaiveBayes': GaussianNB(),
            'KNeighbors': KNeighborsClassifier(n_neighbors=3),
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=10, max_depth=5),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=10, max_depth=3),
            # 'LightGBM': LGBMClassifier(),
            'CatBoost': CatBoostClassifier(silent=True, allow_writing_files=False, iterations=10, depth=3)
        }

        # DataFrame to store metrics
        metrics_list = []

        plot_params = {}
        # Apply each model
        for i, item in enumerate(models.items()):
            try:
                name, model = item
                print(f"Fitting {name}")

                # Fit the model
                model.fit(X_train, y_train)

                # Make predictions on test set
                y_pred = model.predict(X_test)
                y_score = model.predict_proba(X_test)  # n * n_classes

                # Make predictions on train set
                y_train_pred = model.predict(X_train)
                y_train_score = model.predict_proba(X_train)

                # Calculate test metrics
                accuracy = accuracy_score(y_test, y_pred)
                if self.task == 'binary_classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_score[:, 1], multi_class='ovr', average='macro')
                else:
                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')

                # Calculate ROC train metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                if self.task == 'binary_classification':
                    train_precision = precision_score(y_train, y_train_pred)
                    train_recall = recall_score(y_train, y_train_pred)
                    train_f1 = f1_score(y_train, y_train_pred)
                    train_roc_auc = roc_auc_score(y_train, y_train_score[:, 1], multi_class='ovr', average='macro')
                else:
                    train_precision = precision_score(y_train, y_train_pred, average='macro')
                    train_recall = recall_score(y_train, y_train_pred, average='macro')
                    train_f1 = f1_score(y_train, y_train_pred, average='macro')
                    train_roc_auc = roc_auc_score(y_train, y_train_score, multi_class='ovr', average='macro')

                # Append results to the metrics list
                metrics_list.append({
                    'Target': target_column, 
                    'Model': name, 
                    "Accuracy": accuracy, 
                    'Precision': precision, 
                    'Recall': recall, 
                    'F1': f1, 
                    'ROC AUC': roc_auc, 
                    'Train Accuracy': train_accuracy, 
                    'Train Precision': train_precision, 
                    'Train Recall': train_recall, 
                    'Train F1': train_f1, 
                    'Train ROC AUC': train_roc_auc
                })


                ########################################
                # Handles unseen target classes 
                # The next few lines ensures the first few columns of 
                # y_test_bin matches the order in y_score. This is important in 
                # plotting multi-class precision-recall.
                ######################################
                # Unique elements in y_train
                unique_y_train = np.unique(y_train)
                # Unique elements in y_test that are not in y_train
                n_classes = len(np.unique(y_test))
                unique_y_test_not_in_train = np.setdiff1d(np.unique(y_test), unique_y_train)
                # Concatenating the two arrays
                combined_unique_classes = np.concatenate([unique_y_train, unique_y_test_not_in_train])

                # Adjust the shape of y_score
                if len(np.unique(y_test)) > y_score.shape[1]:
                    # Initialize an array to hold the new y_score with columns for unseen classes
                    zeros_to_append = np.zeros((y_score.shape[0], len(combined_unique_classes) - y_score.shape[1]))
                    y_score = np.hstack([y_score, zeros_to_append])

                y_test_bin = label_binarize(y_test, classes=combined_unique_classes)
                #print("n_classes:",n_classes, "y_test_bin.shape:",y_test_bin.shape, "y_score.shape:",y_score.shape)

                plot_params[name] = {"y_test_bin":y_test_bin, "y_score":y_score}

            except Exception as e:
                print(f"Exception {e} occured when fitting utility model {name}! Skipping. ")
                continue

        # Convert metrics list to DataFrame
        metrics = pd.DataFrame(metrics_list)
        
        # Return metrics and figures
        return metrics, plot_params
    
    def calculate_value(self, *args, **kwargs):
            pass