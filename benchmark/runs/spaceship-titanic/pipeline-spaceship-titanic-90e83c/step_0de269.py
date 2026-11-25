import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")
print(train.isnull().sum())
print(train['Transported'].value_counts())
