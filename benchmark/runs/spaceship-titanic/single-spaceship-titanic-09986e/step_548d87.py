import pandas as pd

train = pd.read_csv('train.csv')
print("Shape:", train.shape)
print("\nTypes:\n", train.dtypes)
print("\nMissing values:\n", train.isnull().sum())
print("\nTarget distribution:\n", train['Transported'].value_counts(normalize=True))
print("\nHead:\n", train.head(3))
