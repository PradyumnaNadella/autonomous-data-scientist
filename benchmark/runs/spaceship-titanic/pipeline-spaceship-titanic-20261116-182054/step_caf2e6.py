import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nMissing values (Train):\n", train.isnull().sum()[train.isnull().sum() > 0])
print("\nTarget distribution:\n", train['Transported'].value_counts(normalize=True))
print("\nColumn types:\n", train.dtypes.value_counts())
print("\nSample rows:\n", train.head(2))
