import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
preds = pd.read_csv('predictions.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Predictions shape: {preds.shape}")
print(f"Columns in predictions: {preds.columns.tolist()}")
print(f"Head of predictions:\n{preds.head()}")
