import pandas as pd
train = pd.read_csv('train.csv')
from features import build_features
X = build_features(train.drop(columns=['Transported']))
# Num is object because it comes from 'split' which produces strings.
print(X['Num'].head())
# The plan said: "treat ... Num as numerical."
# Currently, it is being treated as categorical (object).
# This violates the plan.
