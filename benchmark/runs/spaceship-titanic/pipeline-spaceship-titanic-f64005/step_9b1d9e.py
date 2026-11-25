import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)

# Check dtypes
print(train.dtypes)
print(train['CryoSleep'].value_counts(dropna=False))
print(train['VIP'].value_counts(dropna=False))
