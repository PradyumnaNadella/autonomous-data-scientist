import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train columns: {train.columns}")
print(f"Test columns: {test.columns}")

from features import build_features
train_f = build_features(train)
test_f = build_features(test)

print(f"Num of columns train_f: {len(train_f.columns)}")
print(f"Num of columns test_f: {len(test_f.columns)}")
print(f"Missing in test_f: {test_f.isnull().sum().sum()}")
