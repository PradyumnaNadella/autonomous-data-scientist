import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
X_raw = build_features(train_df.drop('Transported', axis=1))

print("Types:")
print(X_raw.dtypes)
print("\nSample values:")
print(X_raw.head())
