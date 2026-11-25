import pandas as pd

code = """
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Cabin parsing
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Num'] = cabin[1]
    df['Side'] = cabin[2]
    
    # PassengerId parsing
    pid = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = pid[0]
    df['Index'] = pid[1]
    
    # Drop original complex columns
    df = df.drop(['Cabin', 'Name'], axis=1)
    
    return df
"""

with open('features.py', 'w') as f:
    f.write(code)

# Test the features
from features import build_features
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train shape:", train_feat.shape)
print("Test shape:", test_feat.shape)
print("Train columns:", train_feat.columns.tolist())
print("Train dtypes:\n", train_feat.dtypes)
