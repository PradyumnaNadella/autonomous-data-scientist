import pandas as pd
train_df = pd.read_csv('train.csv')
from features import build_features
train_df = build_features(train_df)
print(f"Total rows: {len(train_df)}")
print(f"Unique Surnames: {train_df['Surname'].nunique()}")
print(f"Top 5 Surnames by count: {train_df['Surname'].value_counts().head(5)}")
