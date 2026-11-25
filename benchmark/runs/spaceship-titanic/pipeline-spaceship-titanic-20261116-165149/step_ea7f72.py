import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(f"Shape: {predictions.shape}")
print(f"Types:\n{predictions.dtypes}")
# Verify if the values are boolean as in train
print(f"Unique values: {predictions['Transported'].unique()}")
