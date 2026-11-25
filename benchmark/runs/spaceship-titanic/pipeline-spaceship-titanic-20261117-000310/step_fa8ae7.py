import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features
train_df = build_features(train)
test_df = build_features(test)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Columns:", train_df.columns.tolist())
print(train_df.head())
