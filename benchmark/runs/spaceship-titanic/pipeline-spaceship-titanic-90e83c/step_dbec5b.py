import pandas as pd
# The issue might be related to imports or local environment, 
# let's try a very minimal write without imports inside the script first if needed.
# Actually, the error might be specific to my internal runner.
# I will output the file content clearly.

features_code = """
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
"""

with open('features.py', 'w') as f:
    f.write(features_code)
