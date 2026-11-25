import pandas as pd
from features import build_features

# Re-check the logic in train_model.py
# The code drops 'Group', 'Num', 'Surname'.
# But it does not treat CryoSleep or VIP as categorical properly in the pipeline
# because it converts to string manually BEFORE the pipeline.

train_df = pd.read_csv('train.csv')
X_train = build_features(train_df.drop('Transported', axis=1))

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        print(f"Col {col} is object")
    elif X_train[col].dtype == 'bool':
        print(f"Col {col} is bool")
    elif X_train[col].dtype == 'float64':
        print(f"Col {col} is numeric")

# Look at CryoSleep
print("CryoSleep type:", X_train['CryoSleep'].dtype)
