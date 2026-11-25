import pandas as pd
train = pd.read_csv('train.csv')
print(train.dtypes)
print(train.isnull().sum())
