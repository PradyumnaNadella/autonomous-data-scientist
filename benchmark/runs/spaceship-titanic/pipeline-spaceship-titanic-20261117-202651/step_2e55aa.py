import pandas as pd
from features import build_features

# Load data
train = pd.read_csv('train.csv')
X_train = build_features(train.drop(columns=['Transported']))
print(X_train.dtypes)
