import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train columns: {train.columns}")
print(f"Test columns: {test.columns}")
print(f"Num unique PassengerId: {len(test['PassengerId'].unique())} / {len(test)}")
# Check for leakage
print(f"Are there any PassengerId in test also in train? {len(set(train['PassengerId']).intersection(set(test['PassengerId'])))}")
