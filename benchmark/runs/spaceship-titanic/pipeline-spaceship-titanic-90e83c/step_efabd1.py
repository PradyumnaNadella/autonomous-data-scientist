import pandas as pd
import numpy as np

# Create features.py content
features_code = """
import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    # Cabin parsing
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    # Drop Cabin
    df = df.drop(columns=['Cabin'])
    
    # Billing
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    
    # Simple ID parsing (Group)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    
    # Drop Name and PassengerId (Name has high cardinality)
    df = df.drop(columns=['Name', 'PassengerId'])
    return df
"""
with open('features.py', 'w') as f:
    f.write(features_code)

# Verify loading
from features import build_features
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Features built successfully.")
print(train_feat.columns)
