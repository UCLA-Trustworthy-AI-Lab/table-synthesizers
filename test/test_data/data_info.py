import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx]

def load_and_process_data():
    datasets = [load_iris, load_wine, load_breast_cancer]
    dataloaders = []
    data_infos = []

    for dataset_loader in datasets:
        dataset = dataset_loader()
        df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        target = dataset.target
        df['target'] = target

        # Infer column types (assume only two types: continuous or categorical)
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Initialize transformers
        minmax_scaler = MinMaxScaler()
        onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # Transform continuous columns
        df[continuous_cols] = minmax_scaler.fit_transform(df[continuous_cols])

        # Transform categorical columns
        if categorical_cols:
            encoded_cats = onehot_encoder.fit_transform(df[categorical_cols])
            encoded_cat_df = pd.DataFrame(encoded_cats, columns=onehot_encoder.get_feature_names_out(categorical_cols))
            df = df.drop(columns=categorical_cols).join(encoded_cat_df)

        # Create data_info dictionary
        data_info = {
            'transform_info': {},
            'encoded_width': df.shape[1],
            "original_size": len(df), 
        }

        start_idx = 0
        for col in df.columns:
            if col in continuous_cols:
                original_dtype = 'continuous'
                transformed_dtypes = {'minmax_scaled': 'continuous'}
                empirical_dist = []
                end_idx = start_idx + 1
            else:
                original_dtype = 'categorical'
                transformed_dtypes = {cat: 'binary' for cat in onehot_encoder.get_feature_names_out([col])}
                end_idx = start_idx + len(transformed_dtypes) - 1
                empirical_dist = list(df[col].value_counts() / len(df))            

            data_info['transform_info'][col] = {
                'original_dtype': original_dtype,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'transformed_dtypes': transformed_dtypes,
                "empirical_dist":empirical_dist
            }

            start_idx = end_idx

        # Convert transformed DataFrame to tensor and then DataLoader
        tensor_data = torch.tensor(df.values, dtype=torch.float32)
        custom_dataset = CustomTensorDataset(tensor_data)
        dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

        dataloaders.append(dataloader)
        data_infos.append(data_info)

    return dataloaders, data_infos

# Example usage
if __name__ == "__main__":
    dataloaders, data_infos = load_and_process_data()
    for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
        print(f"Dataset {i + 1} Data Info:", data_info)
        for batch in dataloader:
            print(batch.shape)
            break  # Print only the first batch for brevity
