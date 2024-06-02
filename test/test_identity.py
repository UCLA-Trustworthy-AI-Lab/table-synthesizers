import sys
import os

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from stg.tableSynthesizer import TableSynthesizer
from test_data.data_info import load_and_process_data

model = 'Identity'
config = {"bootstrap":False}

print("Test identity return:")
dataloaders, data_infos = load_and_process_data()
for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
    synthesizer = TableSynthesizer(model, config, data_info)
    synthesizer.fit(dataloader)
    sampled_data = synthesizer.sample(n=len(dataloader))
    print("*"*40)
    print(sampled_data.shape, type(sampled_data),len(dataloader))
    print("*"*40)


config = {"bootstrap":True}
print("Test bootstrap:")
dataloaders, data_infos = load_and_process_data()
for i, (dataloader, data_info) in enumerate(zip(dataloaders, data_infos)):
    synthesizer = TableSynthesizer(model, config, data_info)
    synthesizer.fit(dataloader)
    sampled_data = synthesizer.sample(n=123)
    print("*"*40)
    print(sampled_data.shape, type(sampled_data),len(dataloader))
    print("*"*40)