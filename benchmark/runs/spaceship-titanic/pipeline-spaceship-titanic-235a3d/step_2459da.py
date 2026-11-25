import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
print("\nColumn Info:")
print(train.dtypes)
print("\nMissing Values (Train):\n", train.isnull().sum()[train.isnull().sum() > 0])
print("\nTarget Distribution:\n", train['Transported'].value_counts(normalize=True))

print("\nSample Data (First 2 rows):")
print(train.head(2))

# Check for potential ID-like columns
# PassengerId is usually GGGG_PP
# Cabin is Deck/Num/Side
