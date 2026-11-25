import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train target distribution:\n{train['Transported'].value_counts(normalize=True)}")
print(f"\nMissing values (Train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")
print(f"\nMissing values (Test):\n{test.isnull().sum()[test.isnull().sum() > 0]}")

print(f"\nColumns: {train.columns.tolist()}")
print(f"\nSample data:\n{train.head(3)}")
