import pandas as pd
from features import build_features

train_df = build_features(pd.read_csv('train.csv'))
print(train_df.dtypes)
print(train_df.head())
