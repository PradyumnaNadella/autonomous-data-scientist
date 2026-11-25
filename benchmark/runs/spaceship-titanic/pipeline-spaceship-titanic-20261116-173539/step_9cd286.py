import pandas as pd
train = pd.read_csv('train.csv')
print(train['PassengerId'].head())
# The team extracted 'Group' as: df['PassengerId'].str.split('_').str[0].astype(int)
# This treats the group identifier as a continuous number.
# This is a violation of the rule "Do not treat ID-like columns... as numeric quantities."
