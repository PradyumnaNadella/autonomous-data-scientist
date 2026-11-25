import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Verify the features
train_feat = build_features(train.drop(columns=['Transported']))
print(f"Features created: {train_feat.columns.tolist()}")
print(train_feat.dtypes)
