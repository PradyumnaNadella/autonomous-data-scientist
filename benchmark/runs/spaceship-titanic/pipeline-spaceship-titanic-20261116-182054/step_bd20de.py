import pandas as pd
train = pd.read_csv('train.csv')
print("PassengerId example:", train['PassengerId'].iloc[0])
print("Cabin example:", train['Cabin'].dropna().iloc[0])
print("\nUnique count in PassengerId:", train['PassengerId'].nunique())
print("Unique count in Cabin:", train['Cabin'].nunique())
print("\nAny overlap in PassengerId between train and test?")
test = pd.read_csv('test.csv')
print(len(set(train['PassengerId']).intersection(set(test['PassengerId']))))
