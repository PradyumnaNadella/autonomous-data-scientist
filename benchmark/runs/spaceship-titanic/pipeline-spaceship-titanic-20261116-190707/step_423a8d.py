import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
print("Features created:")
print(X.columns)
print("Numeric columns in X:")
print(X.select_dtypes(include=['number']).columns)
