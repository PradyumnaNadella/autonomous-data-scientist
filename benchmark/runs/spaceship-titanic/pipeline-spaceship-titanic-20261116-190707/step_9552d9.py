import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Features generated:")
print(train_feat.columns.tolist())
print(train_feat.dtypes)
