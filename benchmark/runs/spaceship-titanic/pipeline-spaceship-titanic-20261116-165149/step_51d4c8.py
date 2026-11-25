# Check the predictions.csv for structure
import pandas as pd
preds = pd.read_csv('predictions.csv')
print(preds.head())
print(preds.columns)
print(preds.shape)
