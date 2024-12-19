import sys
import os
import torch
import pickle
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
    config_path = os.getenv('SYNTHESIZER_CONFIG_PATH')
    transformed_data_path = os.getenv('TRANSFORMED_DATA_PATH')
    output_path = os.getenv('SYNTHESIZER_OUTPUT_PATH')

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
        tensor_data = torch.tensor(df.values, dtype=torch.float32)
        custom_dataset = CustomTensorDataset(tensor_data)
        dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
        print("DataLoader created successfully from CSV.")
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

    synthesizer.fit(dataloader)
    sampled_data = synthesizer.sample(n=N)

    # Convert sampled_data tensor to CSV
    column_names = []
    for col, param in checkpoint['transformers_parameters'].items():
        column_names += [col+surfix for surfix in param['transformed_surfixes']]
    if isinstance(sampled_data, torch.Tensor):
        sampled_data = sampled_data.numpy()
    print(sampled_data.shape)
    df = pd.DataFrame(sampled_data, columns=column_names)
    synthetic_data_file = os.path.join(output_path, 'synthetic_data.csv')
    df.to_csv(synthetic_data_file, index=False)
    print(f"Synthetic data saved to {synthetic_data_file}")

if __name__ == "__main__":
    main()
