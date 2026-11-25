import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PassengerId: gggg_pp -> Group, IdInGroup
    df[['Group', 'IdInGroup']] = df['PassengerId'].str.split('_', expand=True)
    
    # Cabin: Deck/Num/Side
    # Need to handle missing values by filling with a placeholder before splitting
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    
    # Drop Name, Cabin
    df = df.drop(columns=['Name', 'Cabin'])
    
    return df

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train columns:", train_feat.columns.tolist())
print("Train dtypes:\n", train_feat.dtypes)
print("\nTest columns:", test_feat.columns.tolist())
print("Test dtypes:\n", test_feat.dtypes)

open('features.py', 'w').write('''
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['Group', 'IdInGroup']] = df['PassengerId'].str.split('_', expand=True)
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Name', 'Cabin'])
    return df
''')
