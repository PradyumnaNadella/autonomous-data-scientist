
import pandas
def build_features(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    return df.drop(columns=['Cabin'])
