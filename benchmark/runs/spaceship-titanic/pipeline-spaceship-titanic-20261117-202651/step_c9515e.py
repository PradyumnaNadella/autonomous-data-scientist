import pandas as pd

def build_features(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.copy()
    
    # Cabin parsing
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Num'] = pd.to_numeric(cabin[1], errors='coerce')
    df['Side'] = cabin[2]
    
    # ID parsing (as categorical, not numeric)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    
    # Spending aggregation
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spending_cols].sum(axis=1)
    
    # Drop raw cabin, keep others
    df = df.drop(columns=['Cabin'])
    
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Columns:", train_feat.columns.tolist())
print("Dtypes:\n", train_feat.dtypes)
