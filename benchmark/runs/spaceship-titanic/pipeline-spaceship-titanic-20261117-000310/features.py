
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    return df.drop(columns=['Cabin', 'Name'])
