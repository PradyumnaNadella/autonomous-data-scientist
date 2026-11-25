
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PassengerId: GGGG_PP -> Group
    if 'PassengerId' in df.columns:
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    # Cabin: Deck/Num/Side
    if 'Cabin' in df.columns:
        cabin = df['Cabin'].fillna('nan/nan/nan')
        df['Deck'] = cabin.apply(lambda x: x.split('/')[0])
        df['CabinNum'] = cabin.apply(lambda x: x.split('/')[1])
        df['Side'] = cabin.apply(lambda x: x.split('/')[2])
        df.loc[df['Deck'] == 'nan', ['Deck', 'CabinNum', 'Side']] = None
        df = df.drop('Cabin', axis=1)

    # Name: Extract Surname
    if 'Name' in df.columns:
        df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notnull(x) else None)
        df = df.drop('Name', axis=1)
        
    return df
