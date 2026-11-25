import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"\nMissing values (train):\n{train.isnull().sum()}")
print(f"\nTarget distribution:\n{train['Transported'].value_counts(normalize=True)}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nSample data:\n{train.head(3)}")
