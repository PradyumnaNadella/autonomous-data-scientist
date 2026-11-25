import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"Train columns: {train_df.columns}")
print(f"Test columns: {test_df.columns}")
# Checking if target leaked in features.py/train_model.py
# The features.py function is called with: train_df.drop('Transported', axis=1)
# That looks correct.
