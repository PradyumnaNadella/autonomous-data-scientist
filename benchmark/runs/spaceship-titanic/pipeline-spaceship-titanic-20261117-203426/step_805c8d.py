import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
print(f"\nTarget distribution:\n{train['Transported'].value_counts(normalize=True)}")
print(f"\nMissing values (Train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")

print("\nColumn types:")
print(train.dtypes)

# Check for combined columns
print("\nSample values for potential combined columns:")
print(train[['PassengerId', 'Cabin', 'Name']].head())
