import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Verify build_features output
train_feats = build_features(train.drop('Transported', axis=1))
print(train_feats.head())
print(train_feats.info())
