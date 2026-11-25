
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'PassengerId' in df.columns:
        df[['Group', 'Member']] = df['PassengerId'].str.split('_', expand=True)
    if 'Cabin' in df.columns:
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df = df.drop(columns=['Cabin'])
    if 'Name' in df.columns:
        df['Surname'] = df['Name'].str.split(' ').str[-1]
        df = df.drop(columns=['Name'])
    return df
