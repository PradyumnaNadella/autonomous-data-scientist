import pandas as pd
train = pd.read_csv('train.csv')
print(train['PassengerId'].head())
# The ID is like '0001_01'. 
# build_features says: df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
# Is 'Group' treated as numeric? 
# In build_features it is not cast to int. 
# In train_model.py:
# numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
# categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
# If 'Group' is an object, it goes into categorical_features.
# Let's check the types.
from features import build_features
X = build_features(train.drop(columns=['Transported']))
print(X.dtypes)
