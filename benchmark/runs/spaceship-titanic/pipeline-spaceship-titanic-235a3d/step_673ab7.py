# Verify predictions.csv exists
import pandas as pd
df = pd.read_csv('predictions.csv')
print(df.head())
