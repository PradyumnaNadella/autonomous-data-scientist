import pandas as pd
train = pd.read_csv('train.csv')
print(train['Transported'].value_counts(normalize=True))
