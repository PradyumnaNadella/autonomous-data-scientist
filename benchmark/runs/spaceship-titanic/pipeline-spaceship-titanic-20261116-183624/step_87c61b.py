import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"Train columns: {train_df.columns}")
print(f"Test columns: {test_df.columns}")
print(f"Missing in train: {train_df.isnull().sum().sum()}")
print(f"Missing in test: {test_df.isnull().sum().sum()}")
