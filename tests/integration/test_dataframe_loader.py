import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): The DataFrame to load data from.
        """
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the row to fetch.
        
        Returns:
            pd.Series: The row of the DataFrame at the given index.
        """
        return self.dataframe.iloc[idx]

def collate_fn(batch):
    """
    Custom collate function to handle batches of pandas Series.
    
    Args:
        batch (list of pd.Series): List of Series to collate.
    
    Returns:
        pd.DataFrame: A DataFrame containing the batch.
    """
    return pd.DataFrame(batch)

# Sample DataFrame
data = {
    'A': range(1, 11),
    'B': range(11, 21),
    'C': range(21, 31)
}
df = pd.DataFrame(data)

# Creating dataset instance
dataframe_dataset = DataFrameDataset(df)

# Creating DataLoader with custom collate function
dataframe_loader = DataLoader(dataframe_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Iterating over DataLoader
print("DataFrame DataLoader Output:")
for batch in dataframe_loader:
    print(batch)
    print()
