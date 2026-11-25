import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Cabin parsing: Deck (Cat), Num (Num), Side (Cat)
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Num'] = pd.to_numeric(cabin[1], errors='coerce')
    df['Side'] = cabin[2]
    
    # ID parsing: Group (categorical)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    
    # Spending aggregation
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spending_cols].fillna(0).sum(axis=1)
    
    # Drop original cabin, keep Name/PassengerId/Transported
    df = df.drop(columns=['Cabin'])
    
    return df

# Save function to file
with open('features.py', 'w') as f:
    f.write("""
import pandas as pd
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Num'] = pd.to_numeric(cabin[1], errors='coerce')
    df['Side'] = cabin[2]
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spending_cols].fillna(0).sum(axis=1)
    return df.drop(columns=['Cabin'])
""")

# Validate
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Columns:", train_feat.columns.tolist())
print("Dtypes:\n", train_feat.dtypes)
