
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PassengerId: Extract group ID
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    # Cabin: Split into Deck, Num, Side
    cabin_split = df['Cabin'].fillna('Unknown/Unknown/Unknown').str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    # Drop Name and original Cabin
    df = df.drop(columns=['Name', 'Cabin'])
    # Treat Num as category (or just non-numeric)
    df['Num'] = df['Num'].replace('Unknown', None)
    return df
