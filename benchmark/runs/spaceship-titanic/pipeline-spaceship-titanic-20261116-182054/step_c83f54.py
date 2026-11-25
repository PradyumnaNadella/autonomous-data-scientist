import pandas as pd
import json

# Check the experiment log
with open('experiments.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

for log in logs:
    print(log)

# Check predictions.csv
sub = pd.read_csv('predictions.csv')
print(sub.head())
print(sub.dtypes)
