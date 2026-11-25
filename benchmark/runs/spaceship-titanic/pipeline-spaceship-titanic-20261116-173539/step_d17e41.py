import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(f"Shape: {predictions.shape}")
print(f"Columns: {predictions.columns}")
test = pd.read_csv('test.csv')
print(f"Test size: {len(test)}")
