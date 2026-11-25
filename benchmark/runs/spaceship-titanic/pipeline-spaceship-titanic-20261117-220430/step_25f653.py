import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train columns: {train.columns}")
print(f"Test columns: {test.columns}")
# Checking Name column
print(f"Name in Train: {'Name' in train.columns}")
print(f"Name in Test: {'Name' in test.columns}")
