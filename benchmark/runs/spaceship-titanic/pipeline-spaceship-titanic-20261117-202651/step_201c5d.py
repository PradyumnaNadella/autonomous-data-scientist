import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
print("\nColumn Types:")
print(train.dtypes.value_counts())

print("\nMissing Values (Train):")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nTarget Distribution:")
print(train['Transported'].value_counts(normalize=True))

print("\nSample Data:")
print(train.head(3))
