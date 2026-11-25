import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Unique PassengerId head:", train['PassengerId'].head().tolist())
print("Unique Cabin head:", train['Cabin'].head().tolist())

# Check if Train and Test have the same columns
print(f"\nTrain columns: {set(train.columns)}")
print(f"Test columns: {set(test.columns)}")
print(f"Missing in test: {set(train.columns) - set(test.columns)}")
