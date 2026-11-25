import pandas as pd
# Check the labels of 'Transported'
train = pd.read_csv('train.csv')
print(f"Unique train Transported values: {train['Transported'].unique()}")
predictions = pd.read_csv('predictions.csv')
print(f"Unique predictions Transported values: {predictions['Transported'].unique()}")
