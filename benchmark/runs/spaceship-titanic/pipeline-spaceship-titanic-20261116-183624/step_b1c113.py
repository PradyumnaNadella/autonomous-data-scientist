import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(predictions.shape)
test = pd.read_csv('test.csv')
print(test.shape)
