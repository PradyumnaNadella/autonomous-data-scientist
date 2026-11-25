import pandas as pd
train_df = pd.read_csv('train.csv')
print(f"Group column in train: {'Group' in build_features(train_df).columns}")
# Let's see what happened to Group.
processed = build_features(train_df)
print(f"Group unique values: {processed['Group'].nunique()}")
