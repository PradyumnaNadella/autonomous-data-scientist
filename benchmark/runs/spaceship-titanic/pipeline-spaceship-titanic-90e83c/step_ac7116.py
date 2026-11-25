import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    # Group: Extract from PassengerId
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    # Cabin: Split into Deck, Num, Side
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    
    # Surname: Extract from Name
    df['Surname'] = df['Name'].str.split(' ').str[-1]
    
    # Billing
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[billing_cols] = df[billing_cols].fillna(0)
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    
    # Drop columns that are high cardinality/not directly useful
    # PassengerId, Name are strings. Cabin is replaced by split cols.
    df = df.drop(columns=['Name', 'Cabin'])
    
    return df

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print(train_feat.columns)
print(train_feat.head())
