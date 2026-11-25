import pandas as pd
from features import build_features
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_f = build_features(train)
test_f = build_features(test)

print("Categorical features in training:", [c for c in train_f.columns if train_f[c].dtype == 'object'])
# Check if 'Group' is categorical
print("Group unique values (train):", train_f['Group'].nunique())
# Checking if Group is treated as a numeric column by mistake in logic
# Wait, 'Group' is in categorical_features in train_model.py
# Check for data leakage in Group parsing
print(train_f.columns)
