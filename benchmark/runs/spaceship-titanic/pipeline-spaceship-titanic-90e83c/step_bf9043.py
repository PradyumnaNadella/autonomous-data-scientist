import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Extract Group from PassengerId
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    
    # Split Cabin
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    
    # Extract Surname
    df['Surname'] = df['Name'].str.split().str[-1]
    
    # Billing
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_f = build_features(train)
test_f = build_features(test)

print("Train dtypes:\n", train_f.dtypes)
print("\nTest dtypes:\n", test_f.dtypes)

open('features.py', 'w').write('''
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    df['Surname'] = df['Name'].str.split().str[-1]
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    return df
''')
