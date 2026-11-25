import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(predictions.dtypes)
print(predictions['Transported'].unique())
