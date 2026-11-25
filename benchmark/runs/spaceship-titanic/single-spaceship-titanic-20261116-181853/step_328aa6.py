import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTarget distribution:\n", train['Transported'].value_counts(normalize=True))
print("\nMissing values:\n", train.isnull().sum())
print("\nData types:\n", train.dtypes)
