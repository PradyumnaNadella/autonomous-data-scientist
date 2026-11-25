import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Verify features.py works row-wise on both
try:
    train_feat = build_features(train.copy())
    test_feat = build_features(test.copy())
    print("Feature engineering successful on train and test.")
    print("Train features columns:", train_feat.columns.tolist())
    print("Test features columns:", test_feat.columns.tolist())
except Exception as e:
    print(f"Feature engineering failed: {e}")
