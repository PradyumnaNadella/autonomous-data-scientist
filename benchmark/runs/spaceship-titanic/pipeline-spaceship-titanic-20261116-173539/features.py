
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PassengerId: Group is the prefix
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    
    # Cabin: Deck/Num/Side
    # Need to handle missing values later, but splitting first
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    
    # Drop columns that are raw text and not directly usable
    df = df.drop(columns=['Name', 'Cabin'])
    
    return df
