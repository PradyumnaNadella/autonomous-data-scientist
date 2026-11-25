import pandas as pd
train = pd.read_csv('train.csv')
print(train['CryoSleep'].value_counts(dropna=False))
