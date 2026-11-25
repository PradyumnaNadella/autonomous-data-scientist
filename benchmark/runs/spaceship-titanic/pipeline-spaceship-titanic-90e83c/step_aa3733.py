import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    
    # Simple split without filling first to avoid issues, then fill later if needed
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Num'] = cabin[1]
    df['Side'] = cabin[2]
    
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    
    df['Surname'] = df['Name'].str.split().str[-1]
    
    return df.drop(['Name', 'Cabin'], axis=1)

train = pd.read_csv('train.csv')
train_f = build_features(train)
print(train_f.columns)
