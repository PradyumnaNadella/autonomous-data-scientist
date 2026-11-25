import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train columns: {train.columns.tolist()}")
print(f"Test columns: {test.columns.tolist()}")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Verify leakage
# Are there any columns in test not in train (except Transported)?
# No, looks standard.
