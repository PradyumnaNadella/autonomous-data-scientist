import pandas as pd
train = pd.read_csv('train.csv')
print(f"Train columns: {train.columns.tolist()}")

# Check train_model.py's implementation of build_features
from features import build_features
processed_train = build_features(train)
print(f"Processed columns: {processed_train.columns.tolist()}")
print(f"Num column type: {processed_train['Num'].dtype}")

# Inspect if PassengerId was used
print(f"PassengerId in train? {'PassengerId' in processed_train.columns}")

# Inspect if Num is numeric
try:
    print(processed_train['Num'].astype(float))
except Exception as e:
    print(f"Error converting Num: {e}")
