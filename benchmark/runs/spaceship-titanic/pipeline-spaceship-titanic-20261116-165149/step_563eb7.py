import pandas as pd
train = pd.read_csv('train.csv')
from features import build_features
X = build_features(train).drop(columns=['Transported'])
print(X.info())
