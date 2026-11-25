
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Cabin parsing
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Num'] = cabin[1]
    df['Side'] = cabin[2]
    
    # PassengerId parsing
    pid = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = pid[0]
    df['Index'] = pid[1]
    
    # Drop original complex columns
    df = df.drop(['Cabin', 'Name'], axis=1)
    
    return df
