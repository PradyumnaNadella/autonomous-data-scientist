import pandas as pd
from features import build_features

df = pd.read_csv('train.csv')
df = build_features(df)
print(df.dtypes)
print(df.isnull().sum())
