import pandas as pd
from features import build_features
# Check what columns are still boolean
df = pd.read_csv('train.csv')
df = build_features(df)
print(df.dtypes)
print(df.head())
# Look for 'bool' columns
for col in df.columns:
    if df[col].dtype == 'bool':
        print(f"Boolean column found: {col}")
