import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Split PassengerId: GGGG_PP
    df[['Group', 'MemberID']] = df['PassengerId'].str.split('_', expand=True)
    # Split Cabin: Deck/Num/Side
    cabin_split = df['Cabin'].fillna('Unknown/Unknown/Unknown').str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    df['CabinNum'] = cabin_split[1]
    # Name is removed due to high cardinality
    return df.drop(columns=['Name', 'Cabin'])

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train columns:", train_feat.columns.tolist())
print("\nTest columns:", test_feat.columns.tolist())

open('features.py', 'w').write('''import pandas as pd
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['Group', 'MemberID']] = df['PassengerId'].str.split('_', expand=True)
    cabin_split = df['Cabin'].fillna('Unknown/Unknown/Unknown').str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    df['CabinNum'] = cabin_split[1]
    return df.drop(columns=['Name', 'Cabin'])
''')
