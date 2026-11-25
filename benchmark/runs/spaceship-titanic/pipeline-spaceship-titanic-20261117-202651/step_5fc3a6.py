import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Cabin Parsing
    # Cabin format: Deck/Num/Side
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    # ID Parsing (Group)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    
    # Total Spend
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spending_cols].sum(axis=1)
    
    # Drop original complex columns
    df = df.drop(columns=['Cabin', 'Name'], errors='ignore')
    
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train dtypes:")
print(train_feat.dtypes)
print("\nTest dtypes:")
print(test_feat.dtypes)

# Save the feature script
with open('features.py', 'w') as f:
    f.write('''
import pandas as pd
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spending_cols].sum(axis=1)
    df = df.drop(columns=['Cabin', 'Name'], errors='ignore')
    return df
''')
