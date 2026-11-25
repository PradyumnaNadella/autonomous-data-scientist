import pandas as pd
from features import build_features
train_raw = pd.read_csv('train.csv')
train_df = build_features(train_raw)
print(train_df.dtypes)
print(train_df.isnull().sum())
