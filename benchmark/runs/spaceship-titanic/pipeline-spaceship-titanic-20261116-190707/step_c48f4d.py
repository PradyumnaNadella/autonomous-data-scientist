import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)

# Check dtypes
print(train.dtypes)
# It seems some columns are objects containing a mix of types
for col in train.columns:
    print(f"{col}: {train[col].apply(type).unique()}")
