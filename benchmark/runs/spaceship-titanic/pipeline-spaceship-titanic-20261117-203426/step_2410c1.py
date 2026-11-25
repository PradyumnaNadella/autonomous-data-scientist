import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
train = build_features(train_df.copy())
print(train.dtypes)
# Find the culprit: Likely 'CryoSleep' or 'VIP' having mixed bool and str/NaN
print(train['CryoSleep'].unique())
print(train['VIP'].unique())
