import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PassengerId: Split into Group and MemberID
    if 'PassengerId' in df.columns:
        df[['Group', 'MemberID']] = df['PassengerId'].str.split('_', expand=True)
    
    # Cabin: Split into Deck, Num, Side
    if 'Cabin' in df.columns:
        # Fill missing with 'Unknown/Unknown/Unknown' to allow splitting
        cabin_split = df['Cabin'].fillna('U/U/U').str.split('/', expand=True)
        df['Deck'] = cabin_split[0]
        df['Num'] = cabin_split[1]
        df['Side'] = cabin_split[2]
        # Drop original Cabin if preferred, but keep it for now
        
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_f = build_features(train)
test_f = build_features(test)

print("Train columns:", train_f.columns.tolist())
print("Train dtypes:\n", train_f.dtypes)
print("\nTest columns:", test_f.columns.tolist())
print("Test dtypes:\n", test_f.dtypes)

# Save the feature script
with open('features.py', 'w') as f:
    f.write('''
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'PassengerId' in df.columns:
        df[['Group', 'MemberID']] = df['PassengerId'].str.split('_', expand=True)
    if 'Cabin' in df.columns:
        cabin_split = df['Cabin'].fillna('U/U/U').str.split('/', expand=True)
        df['Deck'] = cabin_split[0]
        df['Num'] = cabin_split[1]
        df['Side'] = cabin_split[2]
    return df
''')
