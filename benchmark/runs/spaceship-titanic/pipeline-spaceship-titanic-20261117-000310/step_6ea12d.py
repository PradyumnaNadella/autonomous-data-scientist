import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target distribution:\n{train['Transported'].value_counts(normalize=True)}")
print(f"\nMissing values (Train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")

print("\nColumns and Dtypes:")
print(train.dtypes)

# ID-like/Composite columns
# PassengerId: GroupId_MemberId
# Cabin: Deck/Num/Side
# Name: First Last
print("\nSample Data:")
print(train.head(2))
