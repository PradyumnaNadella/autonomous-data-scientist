import pandas as pd
# Check if 'Group' in train_model.py is indeed a numerical ID
# It is converted to int.
# Let's verify the distribution of PassengerId
train = pd.read_csv('train.csv')
groups = train['PassengerId'].str.split('_').str[0]
print(f"Number of groups: {groups.nunique()}")
print(f"Max group ID: {groups.astype(int).max()}")
# Treat as categorical feature instead of numeric?
