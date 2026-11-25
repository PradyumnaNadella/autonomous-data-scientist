import pandas as pd
train = pd.read_csv('train.csv')
print(train.isnull().sum())
print("-" * 20)
test = pd.read_csv('test.csv')
print(test.isnull().sum())
