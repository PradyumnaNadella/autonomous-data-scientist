import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PassengerId: Group is the part before the underscore
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    # Cabin: format is Deck/Num/Side
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    # Name: Extract surname
    df['Surname'] = df['Name'].str.split().str[-1]
    
    # Drop original columns we've parsed
    return df.drop(columns=['Cabin', 'Name'])

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train shape:", train_feat.shape)
print(train_feat.dtypes)
print("\nTest shape:", test_feat.shape)
print(test_feat.dtypes)

# Save the function
open('features.py', 'w').write('''
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Surname'] = df['Name'].str.split().str[-1]
    return df.drop(columns=['Cabin', 'Name'])
''')
