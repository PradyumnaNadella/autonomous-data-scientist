from features import build_features
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Test if it runs and handles data properly
try:
    train_feat = build_features(train.copy())
    test_feat = build_features(test.copy())
    print("Columns after features:", train_feat.columns.tolist())
    print("Train features shape:", train_feat.shape)
    print("Test features shape:", test_feat.shape)
except Exception as e:
    print(f"Error: {e}")
