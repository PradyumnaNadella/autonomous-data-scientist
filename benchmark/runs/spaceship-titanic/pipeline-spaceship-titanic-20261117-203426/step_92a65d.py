import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Test build_features
try:
    processed_train = build_features(train_df)
    print("Features engineered successfully.")
    print(processed_train.dtypes)
    print(processed_train.head())
except Exception as e:
    print(f"Error in build_features: {e}")
