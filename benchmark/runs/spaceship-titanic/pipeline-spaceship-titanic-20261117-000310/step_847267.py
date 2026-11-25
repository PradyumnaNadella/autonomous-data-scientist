# Check for leakage
import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Verify build_features does not use target
train_feat = build_features(train_df.drop(columns=['Transported']))
# The check is: did it use the target?
# build_features logic is clear, no 'Transported' used.

# Are all categorical features treated as categorical?
print(train_feat.dtypes)
