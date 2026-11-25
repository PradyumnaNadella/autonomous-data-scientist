import pandas as pd
from features import build_features
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_f = build_features(train)
test_f = build_features(test)

# Check if build_features uses target or anything it shouldn't
print(f"Target in training set: {'Transported' in train_f.columns}")
# Look for consistency
print(f"Train columns: {train_f.columns}")
print(f"Test columns: {test_f.columns}")
