import sys
import os
import torch
import pickle
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.stg.tableSynthesizer import TableSynthesizer

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx]

def main():
    parser = argparse.ArgumentParser(description='TVAE example with management system')
    parser.add_argument('--profile', default='quick', choices=['quick', 'default', 'production'],
                       help='Configuration profile to use')
    parser.add_argument('--with-wandb', action='store_true',
                       help='Enable WandB experiment tracking')
    parser.add_argument('--template', type=str, default=None,
                       help='Configuration template to apply')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to the config file')
    parser.add_argument('--transformed-data-path', type=str, default=None,
                       help='Path to the transformed data')
    parser.add_argument('--synthesizer-output-path', type=str, default=None,
                       help='Path to the synthesizer output')
    args = parser.parse_args()

    print("="*80)
    print("TABLE SYNTHESIZERS - TVAE Example with Management System")
    print("="*80)

    # Prioritize args, fallback to env vars
    config_path = args.config_path or os.environ.get('SYNTHESIZER_CONFIG_PATH')
    transformed_data_path = args.transformed_data_path or os.environ.get('TRANSFORMED_DATA_PATH')
    output_path = args.synthesizer_output_path or os.environ.get('SYNTHESIZER_OUTPUT_PATH')

    if not config_path:
        raise ValueError("Config path must be provided via --config-path or SYNTHESIZER_CONFIG_PATH env var")
    if not transformed_data_path:
         raise ValueError("Transformed data path must be provided via --transformed-data-path or TRANSFORMED_DATA_PATH env var")
    if not output_path:
        raise ValueError("Output path must be provided via --synthesizer-output-path or SYNTHESIZER_OUTPUT_PATH env var")
    if not all([config_path, transformed_data_path, output_path]):
        raise ValueError("Environment variables SYNTHESIZER_CONFIG_PATH, TRANSFORMED_DATA_PATH, and SYNTHESIZER_OUTPUT_PATH must be set.")

    # Load the config JSON
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = config.get('model')
        N = config.get('N')
        if model is None or N is None:
            raise ValueError("The config file must contain 'model' and 'N' keys.")
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    data_csv_file = os.path.join(transformed_data_path, 'transform_output.csv')
    checkpoint_file = os.path.join(transformed_data_path, 'checkpoint.json')
    data_info_file = os.path.join(transformed_data_path, 'data_info.json')

    # Load the DataFrame from CSV
    if os.path.exists(data_csv_file):
        df = pd.read_csv(data_csv_file)
        
        # Models that require raw DataFrame input (e.g., TabSyn)
        # Add other DataFrame-preferring models to this list as needed
        DATAFRAME_MODELS = ['tabsyn', 'tabsynsynthesizer'] 
        
        # Normalize model name for comparison
        model_key = model.lower() if isinstance(model, str) else ''
        
        if model_key in DATAFRAME_MODELS:
            print(f"Model '{model}' detected: Using DataFrame input directly.")
            # Verify DataFrame content before passing
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            training_data = df
        else:
            # Legacy behavior: Convert to Tensor DataLoader
            print(f"Model '{model}' detected: Converting to Tensor DataLoader.")
            try:
                # Attempt to convert to float tensors - this assumes all data is numerical!
                tensor_data = torch.tensor(df.values, dtype=torch.float32)
                custom_dataset = CustomTensorDataset(tensor_data)
                training_data = DataLoader(custom_dataset, batch_size=32, shuffle=True)
                print("DataLoader created successfully from CSV.")
            except ValueError as e:
                # Catch conversion errors (e.g., strings in data)
                raise ValueError(f"Failed to convert data to FloatTensor for model '{model}'. "
                               f"Ensure all data is numerical or use a model that supports DataFrames. Error: {e}")
    else:
        raise FileNotFoundError(f"Data CSV file not found at {data_csv_file}")

    # Load the Checkpoint
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print("checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_file}")
    
    if os.path.exists(data_info_file):
        with open(data_info_file, 'r') as f:
            data_info = json.load(f)
        print("data_info_file loaded successfully.")
    else:
        raise FileNotFoundError(f"Data info file not found at {data_info_file}")

    synthesizer = TableSynthesizer(model, config, data_info)

    # Use the appropriate training data (DataFrame or DataLoader)
    synthesizer.fit(training_data)
    sampled_data = synthesizer.sample(n=N)

    # Convert sampled_data to DataFrame if needed
    column_names = []
    for col, param in checkpoint['transformers_parameters'].items():
        column_names += [col+surfix for surfix in param['transformed_surfixes']]
    
    if isinstance(sampled_data, torch.Tensor):
        sampled_data = sampled_data.detach().cpu().numpy()
        df_out = pd.DataFrame(sampled_data, columns=column_names)
    elif isinstance(sampled_data, pd.DataFrame):
        df_out = sampled_data
        # Ensure columns match if possible, otherwise use generated names
        if len(df_out.columns) == len(column_names):
            df_out.columns = column_names
    else:
         # Numpy array or other iterable
        df_out = pd.DataFrame(sampled_data, columns=column_names)

    print(f"Output shape: {df_out.shape}")
    synthetic_data_file = os.path.join(output_path, 'synthetic_data.csv')
    df_out.to_csv(synthetic_data_file, index=False)
    print(f"Synthetic data saved to {synthetic_data_file}")

if __name__ == "__main__":
    main()
