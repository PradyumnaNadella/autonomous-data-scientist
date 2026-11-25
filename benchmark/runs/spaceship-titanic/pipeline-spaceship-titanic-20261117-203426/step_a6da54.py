import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PassengerId: group_id
    if 'PassengerId' in df.columns:
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    # Cabin: Deck/Num/Side
    if 'Cabin' in df.columns:
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df.drop(columns=['Cabin'], inplace=True)
    
    # Name: Last name
    if 'Name' in df.columns:
        df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notnull(x) else 'Unknown')
        df.drop(columns=['Name'], inplace=True)
        
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_f = build_features(train)
test_f = build_features(test)

print("Train columns:", train_f.columns.tolist())
print("Train dtypes:\n", train_f.dtypes)
print("-" * 20)
print("Test columns:", test_f.columns.tolist())
print("Test dtypes:\n", test_f.dtypes)

open('features.py', 'w').write('''import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'PassengerId' in df.columns:
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    if 'Cabin' in df.columns:
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df.drop(columns=['Cabin'], inplace=True)
    if 'Name' in df.columns:
        df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notnull(x) else 'Unknown')
        df.drop(columns=['Name'], inplace=True)
    return df
''')
