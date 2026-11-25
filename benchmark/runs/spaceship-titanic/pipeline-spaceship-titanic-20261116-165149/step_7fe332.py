import pandas as pd
from features import build_features

def add_group_size(df):
    df = df.copy()
    # Problem: GroupSize calculation relies on full dataset being passed
    # If build_features is used in the pipeline, it won't have the whole dataset
    # Wait, the current implementation in train_model.py adds GroupSize before the pipeline!
    # Let's check if build_features works fine.
    return df

# Check if build_features correctly handles test.csv
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = build_features(train.drop(columns=['Transported']))
X_test = build_features(test)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_train columns: {X_train.columns}")
print(f"X_test columns: {X_test.columns}")
