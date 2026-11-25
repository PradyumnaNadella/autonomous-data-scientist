import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nMissing values (train):\n", train.isnull().sum())
print("\nTarget distribution:\n", train['Transported'].value_counts(normalize=True))
print("\nColumns:\n", train.columns)
print("\nSample (train):\n", train.head(3))
