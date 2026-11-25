import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions.head())
print(predictions.shape)
print(test_df := pd.read_csv('test.csv'))
print(len(predictions) == len(test_df))
