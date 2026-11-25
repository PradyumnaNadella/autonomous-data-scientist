import pandas as pd
import json

with open('experiments.jsonl', 'r') as f:
    for line in f:
        print(line.strip())

train = pd.read_csv('train.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {pd.read_csv('test.csv').shape}")
