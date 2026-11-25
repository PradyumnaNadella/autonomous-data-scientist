import pandas as pd

def build_features(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.copy()
    
    # Split PassengerId: GGGG_PP
    df[['Group', 'MemberID']] = df['PassengerId'].str.split('_', expand=True)
    
    # Split Cabin: Deck/Num/Side
    # Need to handle missing values as strings first so splitting doesn't fail
    cabin_split = df['Cabin'].fillna('Unknown/Unknown/Unknown').str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    # Keep CabinNum as a feature (Num) - note: it's a string initially
    df['CabinNum'] = cabin_split[1]
    
    # Drop columns that shouldn't be used as direct features
    # Cabin is split. Name is too high cardinality (dropping for now).
    # PassengerId itself is kept for merging but excluded from model features later.
    df = df.drop(columns=['Name', 'Cabin'])
    
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train columns:", train_feat.columns.tolist())
print("Train dtypes:\n", train_feat.dtypes)
print("\nTest columns:", test_feat.columns.tolist())
print("Test dtypes:\n", test_feat.dtypes)

open('features.py', 'w').write('''import pandas
def build_features(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.copy()
    df[['Group', 'MemberID']] = df['PassengerId'].str.split('_', expand=True)
    cabin_split = df['Cabin'].fillna('Unknown/Unknown/Unknown').str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    df['CabinNum'] = cabin_split[1]
    return df.drop(columns=['Name', 'Cabin'])
''')
