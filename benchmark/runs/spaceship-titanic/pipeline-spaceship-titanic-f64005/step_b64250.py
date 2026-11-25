import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)
print(train.dtypes)
print(train.head())
