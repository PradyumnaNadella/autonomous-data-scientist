import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")
print(f"Intersection: {set(train_df.columns).intersection(set(test_df.columns))}")
print(f"Test size: {len(test_df)}")
