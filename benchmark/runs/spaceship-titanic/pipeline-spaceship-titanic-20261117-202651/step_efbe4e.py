# Final check of the submission file structure
import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(predictions.columns)
