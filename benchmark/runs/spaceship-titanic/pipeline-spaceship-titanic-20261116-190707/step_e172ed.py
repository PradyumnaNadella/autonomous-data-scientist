import pandas as pd
df = pd.read_csv('predictions.csv')
df['Transported'] = df['Transported'].astype(bool)
df.to_csv('predictions.csv', index=False)
print(df.dtypes)
