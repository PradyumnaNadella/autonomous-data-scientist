import pandas as pd
train = pd.read_csv('train.csv')
print(train['Transported'].head())
print(train['Transported'].dtype)
