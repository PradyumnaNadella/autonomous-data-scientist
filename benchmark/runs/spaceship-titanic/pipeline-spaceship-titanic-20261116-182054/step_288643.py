import pandas as pd
train_df = pd.read_csv('train.csv')
print("Train Columns:", train_df.columns)
print("Nulls:\n", train_df.isnull().sum())
