import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train columns:", train.columns)
print("Missing values in train:\n", train.isnull().sum())
print("Target distribution:\n", train['Transported'].value_counts(normalize=True))
print("Sample of data:\n", train.head(3))
