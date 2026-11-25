import pandas as pd
train_df = pd.read_csv('train.csv')
print(f"Train columns: {train_df.columns}")
test_df = pd.read_csv('test.csv')
print(f"Test columns: {test_df.columns}")
print(f"Any overlap in PassengerId? {set(train_df['PassengerId']).intersection(set(test_df['PassengerId']))}")
