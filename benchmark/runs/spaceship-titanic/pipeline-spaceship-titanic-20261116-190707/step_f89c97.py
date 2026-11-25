import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Verify build_features output
train_feat = build_features(train)
print(train_feat.info())
print(train_feat.head())
