# Checking columns one last time
import pandas as pd
test = pd.read_csv('test.csv')
sub = pd.read_csv('predictions.csv')
print(sub.head())
print(sub.shape)
print(test.shape)
