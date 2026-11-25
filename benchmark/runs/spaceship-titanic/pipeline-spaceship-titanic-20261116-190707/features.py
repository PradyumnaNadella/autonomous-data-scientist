
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Surname'] = df['Name'].str.split().str[-1]
    return df.drop(columns=['Cabin', 'Name'])
