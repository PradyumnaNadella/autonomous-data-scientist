import pandas as pd
# Check if any ID columns were used in the actual features inside train_model
from features import build_features
train = pd.read_csv('train.csv')
X = build_features(train).drop(columns=['Transported'])
print(f"Features used: {list(X.columns)}")
