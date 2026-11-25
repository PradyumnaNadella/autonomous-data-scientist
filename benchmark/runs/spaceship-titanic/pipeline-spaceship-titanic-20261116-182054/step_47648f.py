import pandas as pd
from features import build_features

df = pd.read_csv('train.csv')
X = build_features(df.drop('Transported', axis=1))

print("Columns types:")
for col in X.columns:
    print(f"{col}: {X[col].dtype}")
