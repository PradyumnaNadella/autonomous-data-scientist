import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nTrain info:")
print(train.info())
print("\nMissing values (Train):\n", train.isnull().sum())
print("\nTarget distribution:\n", train['Transported'].value_counts(normalize=True))

# Identify composite columns
# PassengerId: GroupId_MemberId
# Cabin: Deck/Num/Side
# Name: First Last
print("\nSample rows:")
print(train.head(3))
