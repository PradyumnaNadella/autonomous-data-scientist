import pandas as pd
from features import build_features

# Check for leaks in categorical features (e.g. Surname in train vs test)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_processed = build_features(train)
test_processed = build_features(test)

print("Train columns:", train_processed.columns)
print("Test columns:", test_processed.columns)

# Look at Surname cardinality
print("Train surnames:", train_processed['Surname'].nunique())
print("Test surnames:", test_processed['Surname'].nunique())
