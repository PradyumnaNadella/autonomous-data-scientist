import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(predictions.columns)
print(predictions['Transported'].value_counts())
# Check format
print(predictions['Transported'].dtype)
