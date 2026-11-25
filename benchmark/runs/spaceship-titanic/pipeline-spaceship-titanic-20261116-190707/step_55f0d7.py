import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
features = build_features(train.drop('Transported', axis=1))
print("Feature columns:", features.columns.tolist())
print("Missing values in features:\n", features.isnull().sum())
