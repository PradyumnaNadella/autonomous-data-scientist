
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
