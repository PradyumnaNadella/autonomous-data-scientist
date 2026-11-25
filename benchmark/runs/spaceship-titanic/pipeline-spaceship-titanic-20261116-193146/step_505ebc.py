import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PassengerId: Extract group ID
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    
    # Cabin: Split into Deck, Num, Side
    # Use fillna to handle missing values before split
    cabin_split = df['Cabin'].fillna('Unknown/Unknown/Unknown').str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    
    # Drop Name and original Cabin
    df = df.drop(columns=['Name', 'Cabin'])
    
    # Ensure ID columns aren't numeric
    df['GroupId'] = df['GroupId'].astype(str)
    df['Num'] = df['Num'].replace('Unknown', None)
    
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_f = build_features(train)
test_f = build_features(test)

print("Train columns and dtypes:")
print(train_f.dtypes)
print("\nTest columns and dtypes:")
print(test_f.dtypes)
