import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    
    # PassengerId: XXXX_YY -> Group
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    
    # Cabin: Deck/Num/Side
    # Need to handle missing values before splitting
    df['Cabin'] = df['Cabin'].fillna('Unknown/0/Unknown')
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    # Billing
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[billing_cols] = df[billing_cols].fillna(0)
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    
    # Name: extract surname
    df['Surname'] = df['Name'].str.split().str[-1]
    
    return df.drop(['Name', 'Cabin'], axis=1)

train = pd.read_csv('train.csv')
train_f = build_features(train)
print(train_f.info())
print(train_f.head())
