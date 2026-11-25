import pandas as pd

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
