import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train columns: {list(train.columns)}")
print(f"Test columns: {list(test.columns)}")
print(f"Train size: {train.shape}")
print(f"Test size: {test.shape}")
