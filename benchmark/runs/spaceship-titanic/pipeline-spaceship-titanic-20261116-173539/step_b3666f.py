import pandas as pd
train_df = pd.read_csv('train.csv')
from features import build_features
train_df = build_features(train_df)

# The logic in train_model.py:
# train_df = train_df.drop(columns=['PassengerId', 'Name'], errors='ignore')
# num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# The 'Num' feature is still a string in the DataFrame returned by build_features
# Does select_dtypes(include=['int64', 'float64']) include it? No.
# Does it get imputed/scaled? No.

print(f"Features: {train_df.columns.tolist()}")
cat_features = train_df.select_dtypes(include=['object', 'bool']).columns.tolist()
num_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Cat: {cat_features}")
print(f"Num: {num_features}")
