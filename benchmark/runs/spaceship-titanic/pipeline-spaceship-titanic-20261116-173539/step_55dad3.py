# Look at the train_model.py again:
# train_df = train_df.drop(columns=['PassengerId', 'Name'], errors='ignore')
# Then...
# cat_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
# num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# The 'Num' column remains in cat_features because it's an object (string).
# Wait, it was an object in the DF, so the preprocessor will one-hot encode it as a categorical column!
# 'Num' is numeric (an identifier), not a categorical feature. 
# This is a major error in feature handling.

import pandas as pd
from features import build_features
train = pd.read_csv('train.csv')
train = build_features(train)
train = train.drop(columns=['PassengerId', 'Name'], errors='ignore')
# Note: build_features already drops Name.

# 'Num' is a string.
print(f"Num column unique values: {train['Num'].nunique()}")
print(f"Num column dtype: {train['Num'].dtype}")
