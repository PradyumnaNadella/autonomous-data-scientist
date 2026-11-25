import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
train_feat = build_features(train)
print(train_feat.dtypes)
print(train_feat.head())
