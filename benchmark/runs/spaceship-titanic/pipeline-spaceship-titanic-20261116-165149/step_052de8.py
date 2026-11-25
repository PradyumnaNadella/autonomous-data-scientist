import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PassengerId: Group_Member
    if 'PassengerId' in df.columns:
        df[['Group', 'Member']] = df['PassengerId'].str.split('_', expand=True)
    
    # Cabin: Deck/Num/Side
    if 'Cabin' in df.columns:
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        # Drop Cabin to avoid redundant high-cardinality feature
        df = df.drop(columns=['Cabin'])
    
    # Name: First_Last (Extract Surname)
    if 'Name' in df.columns:
        df['Surname'] = df['Name'].str.split(' ').str[-1]
        df = df.drop(columns=['Name'])
        
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train dtypes:")
print(train_feat.dtypes)
print("\nTest dtypes:")
print(test_feat.dtypes)

open('features.py', 'w').write('''
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
''')
