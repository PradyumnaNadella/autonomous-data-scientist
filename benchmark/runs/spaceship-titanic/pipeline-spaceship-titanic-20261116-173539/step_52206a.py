import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")
print("\nTrain Info:")
print(train.info())

print("\nMissing Values (Train):")
print(train.isnull().sum())

print("\nTarget Distribution (Transported):")
print(train['Transported'].value_counts(normalize=True))

print("\nSample Rows:")
print(train.head(3))
