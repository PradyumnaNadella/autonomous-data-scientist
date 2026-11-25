import pandas as pd
preds = pd.read_csv('predictions.csv')
# XGBClassifier with boolean labels might predict True/False, but if it outputs 0/1, fix it.
# Wait, let's check the values.
print(preds['Transported'].unique())
# The target in train is boolean. If XGB predict is 0/1, convert to boolean.
train = pd.read_csv('train.csv')
print(train['Transported'].dtype)

# Fix
preds['Transported'] = preds['Transported'].astype(bool)
preds.to_csv('predictions.csv', index=False)
print(preds.head())
