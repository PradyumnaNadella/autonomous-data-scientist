import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(train.info())
print(train.head())
print(train['Transported'].value_counts(normalize=True))
print(train.isnull().sum())
