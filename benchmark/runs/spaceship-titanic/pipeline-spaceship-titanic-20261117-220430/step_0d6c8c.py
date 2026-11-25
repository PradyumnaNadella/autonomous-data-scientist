import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(f"Shape: {predictions.shape}")
print(f"Test size: {pd.read_csv('test.csv').shape[0]}")
