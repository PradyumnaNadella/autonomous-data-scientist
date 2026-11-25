import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target distribution:\n{train['Transported'].value_counts(normalize=True)}")
print("\nMissing values:\n", train.isnull().sum()[train.isnull().sum() > 0])

# Check for mixed columns/IDs
print("\nHead of train:")
print(train.head(3))
