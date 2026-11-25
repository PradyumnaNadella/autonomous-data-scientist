import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
train_df_f = build_features(train_df)
print(train_df_f.dtypes)
print(train_df_f.head())
