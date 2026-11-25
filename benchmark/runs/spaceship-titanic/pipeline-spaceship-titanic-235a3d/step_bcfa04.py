import pandas as pd
df_pred = pd.read_csv('predictions.csv')
print(df_pred.columns)
print(df_pred.dtypes)
print(df_pred.head())
