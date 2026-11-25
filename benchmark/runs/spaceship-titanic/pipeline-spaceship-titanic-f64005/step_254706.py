import pandas as pd
import json

# Check predictions.csv
preds = pd.read_csv('predictions.csv')
print("Predictions columns:", preds.columns.tolist())
print("Predictions shape:", preds.shape)
print("Predictions head:\n", preds.head())

# Check experiments.jsonl
with open('experiments.jsonl', 'r') as f:
    for line in f:
        print(line.strip())
