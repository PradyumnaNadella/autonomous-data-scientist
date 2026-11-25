import pandas as pd
train = pd.read_csv('train.csv')
from features import build_features
X = build_features(train).drop(columns=['Transported'])
# PassengerId is still in X, and it's being treated as an object/string
print(X[['PassengerId', 'Group', 'Member']].head())
