import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PassengerId: Extract group
    df['GroupId'] = df['PassengerId'].str.split('_', expand=True)[0]
    
    # Cabin: Split into Deck, Num, Side
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    
    # Expenditures: Total
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[spend_cols].sum(axis=1)
    df['HasSpent'] = (df[spend_cols].sum(axis=1) > 0).astype(int)
    
    # Drop original complex columns
    df = df.drop(columns=['Cabin', 'Name'])
    
    return df

# Test on datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train columns:", train_feat.columns.tolist())
print("Train dtypes:\n", train_feat.dtypes)
print("\nTest columns:", test_feat.columns.tolist())
print("Test dtypes:\n", test_feat.dtypes)

# Save feature function
with open('features.py', 'w') as f:
    f.write('''import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['GroupId'] = df['PassengerId'].str.split('_', expand=True)[0]
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[spend_cols].sum(axis=1)
    df['HasSpent'] = (df[spend_cols].sum(axis=1) > 0).astype(int)
    return df.drop(columns=['Cabin', 'Name'])
''')
