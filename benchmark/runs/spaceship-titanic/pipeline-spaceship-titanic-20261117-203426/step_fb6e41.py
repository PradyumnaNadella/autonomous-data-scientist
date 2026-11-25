import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
train = build_features(train_df.copy())
print(train.info())
print(train.head())
