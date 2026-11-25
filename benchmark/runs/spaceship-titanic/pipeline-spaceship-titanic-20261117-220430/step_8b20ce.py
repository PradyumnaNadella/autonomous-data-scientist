from features import build_features
import pandas as pd
train_df = pd.read_csv('train.csv')
processed = build_features(train_df)
print(f"Group unique values: {processed['Group'].nunique()}")
