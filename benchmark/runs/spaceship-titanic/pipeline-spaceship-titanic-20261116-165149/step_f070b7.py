import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train columns: {train.columns}")
print(f"Test columns: {test.columns}")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
