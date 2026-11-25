import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Column Types:")
print(train.dtypes)

print("\nCabin example:", train['Cabin'].dropna().iloc[0])
print("PassengerId example:", train['PassengerId'].iloc[0])
