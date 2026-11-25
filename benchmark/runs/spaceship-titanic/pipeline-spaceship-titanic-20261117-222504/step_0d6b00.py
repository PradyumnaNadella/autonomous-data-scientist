import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Leakage/Structure
print(f"Columns diff: {set(train.columns) - set(test.columns)}")
print("\nCabin format:", train['Cabin'].dropna().unique()[:3])
print("PassengerId format:", train['PassengerId'].unique()[:3])
