import torch
import itertools
import pandas as pd
import numpy as np
import os
import glob

# Optional imports - only load if available
try:
    from syntest import DataHost
    SYNTEST_AVAILABLE = True
except ImportError:
    SYNTEST_AVAILABLE = False

try:
    import dask.dataframe as dd
    from lddp_optimus import TableTransformer
    LDDP_AVAILABLE = True
except ImportError:
    LDDP_AVAILABLE = False

try:
    import sdmetrics
    from sdmetrics.reports.single_table import QualityReport
    SDMETRICS_AVAILABLE = True
except ImportError:
    SDMETRICS_AVAILABLE = False


# Generate all combinations of the parameter choices
n_num_choices = [5, 0] #0, 5
n_cat_choices = [5, 0] #0, 5
n_rows_choices = [1000]
missing_percent_choices = [0, 0.1] #0, 0.1, 0.5, 0.9

parameter_combinations = list(itertools.product(
    n_num_choices,
    n_cat_choices,
    n_rows_choices,
    missing_percent_choices
))

# Format the combinations into the desired structure
test_parameters = [
    (n_num, n_cat, n_rows, missing_percent)
    for n_num, n_cat, n_rows, missing_percent in parameter_combinations if (n_cat > 0 or n_num > 0)
]


def get_data(data_source):
    """
        Get data and config from syntest
    """
    if not SYNTEST_AVAILABLE:
        raise ImportError("syntest module not available")
        
    datasets = {}
    host = DataHost()
    if data_source in ['real', 'both']:
        datasets.update(host.get_all_datasets())
    if data_source in ['fake', 'both']:
        for n_num, n_cat, n_rows, missing_percent in test_parameters:
            datasets[f'fake_{n_num}_{n_cat}_{n_rows}_{missing_percent}'] = host.get_fake_dataset(n_num, n_cat, n_rows, missing_percent)
    if data_source not in ['real', 'fake','both']:
        real, fake, config = host.get_dataset(data_source)
        datasets[data_source] = {'real':real, 'fake':fake, 'config':config}

    return datasets

def get_model_config(model_name):
    """
        Return model specific mapping + training parameters
    """
    if model_name == 'Identity':
        model_config = {'mapping':{
            "continuous": "mm",
            "categorical": "ohe",
            "pii": "pii",
            "datetime": "dt"
        },"bootstrap": False}
    if model_name == 'CTGAN':
        model_config = {'mapping':{
            "continuous": "mm",
            "categorical": "ohe",
            "pii": "pii",
            "datetime": "dt"
        },"epochs": 300, "batch_size":500}
    if model_name == "TabDDPM":
        model_config = {'mapping':{
            "continuous": "gqt",
            "categorical": "le",
            "pii": "pii",
            "datetime": "dt"
        },"steps": 10000,
        "batch_size":1024}

    return model_config

def transform_data(real, config, mapping, batch_size = 300, **kwargs):
    """
        Transform the data above to tensor loader using lddp
    """
    if not LDDP_AVAILABLE:
        raise ImportError("lddp_optimus module not available")
        
    ddf = dd.from_pandas(real)

    meta = config['meta']
    for c in meta:
        if meta[c] == "numerical":
            meta[c] = 'continuous'

    tableTransformer = TableTransformer(meta, mapping)
    #tableTransformer.set_local_cluaster()
    transformed_ddf = tableTransformer.fit_transform(ddf)
    print("Head of transformed data:")
    print(transformed_ddf.head())

    tensor_loader = tableTransformer.ddf_to_tensor_loader_naive(transformed_ddf, batch_size)


    for batch in tensor_loader:
        print(batch.shape, type(batch))
        break 

    #tableTransformer.close_local_clusters()

    return tensor_loader, tableTransformer.get_transform_info(), tableTransformer


def validate_output(real_df, sampled_data, data_info, config, tableTransformer):
    """
        Confirm the size/type of synthetic data is good. TODO: add automatic quality check.
    """
    if not LDDP_AVAILABLE or not SDMETRICS_AVAILABLE:
        raise ImportError("Required modules not available")
        
    assert sampled_data.shape[0] == data_info['original_size'], "Sampled data length mismatch"
    assert isinstance(sampled_data, torch.Tensor), "Sampled data must be tensor!"

    print(config['meta'])

    sampled_df = tableTransformer.tensor_to_ddf(sampled_data)
    print("sampled_df:")
    print(sampled_df.compute())
    sampled_df.compute().to_csv("sampled_df.csv",index=False)
    synthetic_df = tableTransformer.inverse_transform(sampled_df).compute()
    metadata = {'columns':{}}
    for c in config['meta']:
        metadata['columns'][c] = {}
        metadata['columns'][c]['sdtype'] = 'categorical' if config['meta'][c] == 'categorical' else 'numerical'

    quality_report = QualityReport()
    quality_report.generate(real_df, synthetic_df, metadata)
    print(quality_report.get_details('Column Shapes'))


    print(real_df.head())
    print(synthetic_df.head())


def create_test_dataframe():
    """Create a simple test DataFrame for testing DataFrame input functionality"""
    np.random.seed(42)  # For reproducible tests
    
    df = pd.DataFrame({
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'binary_col': np.random.choice([0, 1], 100),
        'float_col': np.random.uniform(0, 1, 100)
    })
    
    return df


def load_sandbox_datasets():
    """
    Load all CSV files from the sandbox_datasets directory
    
    Returns:
        dict: Dictionary mapping dataset names to pandas DataFrames
    """
    sandbox_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'sandbox_datasets')
    datasets = {}
    
    # Find all CSV files in the sandbox directory
    csv_files = glob.glob(os.path.join(sandbox_dir, '*.csv'))
    
    for csv_file in csv_files:
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
        try:
            # Load CSV with proper handling for different formats
            df = pd.read_csv(csv_file)
            
            # Basic data info
            print(f"Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  Columns: {list(df.columns)}")
            
            # Identify data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            print(f"  Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
            print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
            
            datasets[dataset_name] = df
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            
    return datasets


def get_realistic_test_config(dataset_name, model_name):
    """
    Get realistic test configurations for different dataset-model combinations
    
    Args:
        dataset_name (str): Name of the dataset (insurance, adult, covtype)
        model_name (str): Name of the synthesizer model
        
    Returns:
        dict: Configuration parameters optimized for the dataset-model combination
    """
    # Only include models that are actually registered in the system
    # Note: CTGAN appears to not be registered, so focusing on available models
    base_configs = {
        'Identity': {"epochs": 1, "batch_size": 32},
        'TabDDPM': {"epochs": 10, "batch_size": 64, "num_timesteps": 100},
        'TVAE': {"epochs": 10, "batch_size": 128},
        'SMOTE': {"k_neighbors": 5, "random_state": 42},
        'PATECTGAN': {"epochs": 10, "batch_size": 128},
        'AIM': {"epsilon": 1.0, "delta": 1e-9, "max_iters": 100},
        'CART': {"max_depth": 10},
        'DPCART': {"max_depth": 10, "epsilon": 1.0},
        'BayesianNetwork': {"max_parents": 3},
        'GREAT': {"epochs": 5, "batch_size": 32},  # Reduced for testing
        'ARF': {"n_estimators": 10},  # Reduced for testing
        'NFlow': {"epochs": 10, "batch_size": 64},
        'AutoDiff': {"n_epochs": 10, "diff_n_epochs": 10, "batch_size": 32},
        'TabSyn': {"epochs": 10, "batch_size": 128}
    }
    
    # Dataset-specific adjustments
    dataset_adjustments = {
        'insurance': {
            'SMOTE': {"target_column": "charges"},  # Regression target
        },
        'adult': {
            'SMOTE': {"target_column": "income"},   # Classification target
        },
        'covtype': {
            'SMOTE': {"target_column": "Cover_Type"},  # Classification target
            # Reduce batch sizes for large dataset
            'TabDDPM': {"batch_size": 128},
            'TVAE': {"batch_size": 256},
            'PATECTGAN': {"batch_size": 256}
        }
    }
    
    # Start with base config
    config = base_configs.get(model_name, {"epochs": 10, "batch_size": 32})
    
    # Apply dataset-specific adjustments
    if dataset_name in dataset_adjustments and model_name in dataset_adjustments[dataset_name]:
        config.update(dataset_adjustments[dataset_name][model_name])
        
    return config


def run_sandbox_dataset_test(model_name, dataset_name, config=None, n_samples=50, sample_ratio=0.1):
    """
    Test a synthesizer model on a specific sandbox dataset
    
    Args:
        model_name (str): Name of the synthesizer model
        dataset_name (str): Name of the sandbox dataset
        config (dict): Configuration parameters for the model
        n_samples (int): Number of samples to generate
        sample_ratio (float): Ratio of original data to use for testing (to speed up tests)
        
    Returns:
        bool: True if test passes
    """
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from stg.tableSynthesizer import TableSynthesizer
    
    # Load sandbox datasets
    datasets = load_sandbox_datasets()
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(datasets.keys())}")
    
    df = datasets[dataset_name]
    
    # Sample data for faster testing if dataset is large
    if len(df) > 1000:
        sample_size = min(int(len(df) * sample_ratio), 1000)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} rows from {dataset_name} for testing")
    
    # Use realistic config if none provided
    if config is None:
        config = get_realistic_test_config(dataset_name, model_name)
    
    print(f"Testing {model_name} on {dataset_name} with config: {config}")
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer(model_name, config)
    
    try:
        # Test fitting with DataFrame
        synthesizer.fit(df, batch_size=config.get('batch_size', 32))
        
        # Test tensor output
        sampled_tensor = synthesizer.sample(n=n_samples)
        # Allow small tolerance for models like SMOTE that may generate slightly different counts due to class proportions
        tolerance = 5 if model_name == 'SMOTE' else 0
        assert abs(sampled_tensor.shape[0] - n_samples) <= tolerance, f"Expected ~{n_samples} samples, got {sampled_tensor.shape[0]}"
        assert isinstance(sampled_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(sampled_tensor)}"
        # Note: tensor dimension may be larger than original DataFrame due to categorical encoding
        assert sampled_tensor.shape[1] >= df.shape[1], f"Feature dimension too small: expected at least {df.shape[1]}, got {sampled_tensor.shape[1]}"
        
        # Test DataFrame output
        sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
        # Allow small tolerance for models like SMOTE that may generate slightly different counts due to class proportions
        tolerance = 5 if model_name == 'SMOTE' else 0
        assert abs(sampled_df.shape[0] - n_samples) <= tolerance, f"Expected ~{n_samples} samples, got {sampled_df.shape[0]}"
        assert isinstance(sampled_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(sampled_df)}"
        # Check that same columns exist (order may vary due to encoding/decoding process)
        expected_cols = set(df.columns)
        actual_cols = set(sampled_df.columns)
        assert expected_cols == actual_cols, f"Column names mismatch. Expected: {expected_cols}, Got: {actual_cols}"
        
        print(f"✅ {model_name} test on {dataset_name} passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} test on {dataset_name} failed: {e}")
        raise


def test_dataframe_support(model_name, config=None, n_samples=10):
    """
    Generic test function for DataFrame support that can be used by any synthesizer
    
    Args:
        model_name (str): Name of the synthesizer model
        config (dict): Configuration parameters for the model
        n_samples (int): Number of samples to generate
    
    Returns:
        bool: True if test passes
    """
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from stg.tableSynthesizer import TableSynthesizer
    
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Use default config if none provided
    if config is None:
        config = {"epochs": 1, "batch_size": 32}
    
    # Initialize synthesizer
    synthesizer = TableSynthesizer(model_name, config)
    
    # Test fitting with DataFrame
    synthesizer.fit(df, batch_size=32)
    
    # Test tensor output
    sampled_tensor = synthesizer.sample(n=n_samples)
    assert sampled_tensor.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_tensor.shape[0]}"
    assert isinstance(sampled_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(sampled_tensor)}"
    
    # Test DataFrame output
    sampled_df = synthesizer.sample(n=n_samples, return_dataframe=True)
    assert sampled_df.shape[0] == n_samples, f"Expected {n_samples} samples, got {sampled_df.shape[0]}"
    assert isinstance(sampled_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(sampled_df)}"
    
    # Check column names (more flexible since encoding might change order)
    expected_cols = set(['numeric_col', 'categorical_col', 'binary_col', 'float_col'])
    actual_cols = set(sampled_df.columns)
    assert expected_cols == actual_cols, f"Column names mismatch. Expected: {expected_cols}, Got: {actual_cols}"
    
    print(f"{model_name} DataFrame test passed successfully!")
    return True

