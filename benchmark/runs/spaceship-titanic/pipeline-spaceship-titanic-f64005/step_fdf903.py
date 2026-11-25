import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target distribution:\n{train['Transported'].value_counts(normalize=True)}")
print(f"\nMissing values (Train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")
print(f"\nMissing values (Test):\n{test.isnull().sum()[test.isnull().sum() > 0]}")

# Check for columns combining information
print(f"\nSample of 'PassengerId': {train['PassengerId'].head(3)}")
print(f"Sample of 'Cabin': {train['Cabin'].head(3)}")
