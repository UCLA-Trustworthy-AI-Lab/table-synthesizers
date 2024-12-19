import torch
from torch.utils.data import DataLoader, Dataset
import random

# Define a custom dataset that generates random one-hot vectors
class RandomOneHotDataset(Dataset):
    def __init__(self, num_samples, vector_size):
        self.num_samples = num_samples
        self.vector_size = vector_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        vector = torch.zeros(self.vector_size)
        random_index = torch.randint(0, self.vector_size, (1,))
        vector[random_index] = 1
        return vector

# Condition function to check if the first element is 1
def condition_function(row):
    return row[0] == 1

# Reservoir sampling function
def reservoir_sample(dataloader, sample_size):
    sample = []
    count = 0
    
    for batch in dataloader:
        for row in batch:
            if condition_function(row):
                count += 1
                if len(sample) < sample_size:
                    sample.append(row)
                else:
                    s = random.randint(0, count - 1)
                    if s < sample_size:
                        sample[s] = row
                        
    return torch.stack(sample)  # Convert list of tensors back to a single tensor

# Create a DataLoader with random one-hot vectors
vector_size = 10  # Size of the one-hot vector
num_samples = 10000  # Number of samples in the dataset
batch_size = 64  # Batch size for the dataloader

dataset = RandomOneHotDataset(num_samples, vector_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Sample from the dataloader
sample_size = 100  # Number of samples you want
sampled_data = reservoir_sample(dataloader, sample_size)

# Confirm all the sampled tensors match the specific condition
all_match = all(condition_function(row) for row in sampled_data)

print("All sampled tensors match the specific condition:", all_match)
print("Sampled Data Shape:", sampled_data.shape)
print("Sampled Data:")
print(sampled_data[:5,:])
