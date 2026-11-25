import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
X_train = build_features(train_df.drop('Transported', axis=1))

print("Types in X_train:")
print(X_train.dtypes)
for col in X_train.columns:
    print(f"Col: {col}, Unique: {X_train[col].unique()[:5]}")
