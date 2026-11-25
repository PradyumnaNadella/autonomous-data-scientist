import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nTrain info:")
print(train.info())
print("\nMissing values (Train):\n", train.isnull().sum())
print("\nTarget distribution (Train):\n", train['Transported'].value_counts(normalize=True))

# ID-like and complex columns
print("\nSample of 'PassengerId', 'Cabin', 'Name':")
print(train[['PassengerId', 'Cabin', 'Name']].head())
