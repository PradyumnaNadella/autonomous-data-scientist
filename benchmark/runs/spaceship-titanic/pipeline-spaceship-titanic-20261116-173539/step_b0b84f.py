import pandas as pd
from features import build_features

df = pd.read_csv('train.csv')
df_feat = build_features(df)
print(df_feat.dtypes)
print(df_feat[['Num', 'Group']].head())
