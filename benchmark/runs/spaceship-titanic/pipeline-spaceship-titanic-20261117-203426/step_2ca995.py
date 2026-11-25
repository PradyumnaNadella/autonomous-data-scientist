import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Split PassengerId
    df[['Group', 'MemberId']] = df['PassengerId'].str.split('_', expand=True)
    
    # Split Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    # Extract Surname from Name
    df['Surname'] = df['Name'].str.split().str[-1]
    
    # Drop raw columns that won't be used directly
    # Keep PassengerId and Transported as required
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train columns:", train_feat.columns.tolist())
print("Test columns:", test_feat.columns.tolist())
print("Dtypes:\n", train_feat.dtypes)

with open('features.py', 'w') as f:
    f.write("""
import pandas as pd
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['Group', 'MemberId']] = df['PassengerId'].str.split('_', expand=True)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Surname'] = df['Name'].str.split().str[-1]
    return df
""")
