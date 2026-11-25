import pandas as pd
print("Imports successful")
train = pd.read_csv('train.csv')
print("Read successful")
print(train.columns)
