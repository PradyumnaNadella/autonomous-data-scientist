import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(predictions['Transported'].value_counts())
train = pd.read_csv('train.csv')
print(train['Transported'].value_counts(normalize=True))
