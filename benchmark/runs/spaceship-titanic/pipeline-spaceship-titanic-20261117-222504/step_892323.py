import pandas as pd
import json

with open('experiments.jsonl', 'r') as f:
    for line in f:
        print(line.strip())

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train columns: {train.columns.tolist()}")
print(f"Test columns: {test.columns.tolist()}")

preds = pd.read_csv('predictions.csv')
print(f"Predictions shape: {preds.shape}")
print(f"Predictions sample: {preds.head()}")
