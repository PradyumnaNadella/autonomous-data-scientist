import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PassengerId: Group_Member
    # Treat as categorical/split, not numeric
    df[['GroupId', 'MemberId']] = df['PassengerId'].str.split('_', expand=True)
    
    # Cabin: Deck/Num/Side
    # Parse into components
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1] # Keep as string for now
    df['Side'] = cabin_split[2]
    
    # Drop Name (not useful for generic prediction) and original Cabin
    df = df.drop(columns=['Name', 'Cabin'])
    
    return df

# Test on train/test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train dtypes:")
print(train_feat.dtypes)
print("\nTest dtypes:")
print(test_feat.dtypes)

# Save to file
code = """
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['GroupId', 'MemberId']] = df['PassengerId'].str.split('_', expand=True)
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Name', 'Cabin'])
    return df
"""
open('features.py', 'w').write(code)
