# Verify predictions file
import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(predictions.columns)
print(predictions.shape)
test = pd.read_csv('test.csv')
print(test.shape)
