import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
print("\nColumn Types:\n", train.dtypes.value_counts())
print("\nMissing Values (Train):\n", train.isnull().sum()[train.isnull().sum() > 0])
print("\nTarget Distribution:\n", train['Transported'].value_counts(normalize=True))

# Identify combined columns
print("\nPotential Combined Columns (examples):")
print(train[['PassengerId', 'Cabin', 'Name']].head(2))
