import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nTrain columns and types:")
print(train.dtypes)
print("\nMissing values (Train):\n", train.isnull().sum())
print("\nTarget distribution (Train):\n", train['Transported'].value_counts(normalize=True))
print("\nSample columns:\n", train.head(1).T)
