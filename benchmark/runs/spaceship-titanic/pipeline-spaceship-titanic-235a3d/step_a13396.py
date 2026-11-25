import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PassengerId: GGGG_PP -> Group
    if 'PassengerId' in df.columns:
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    # Cabin: Deck/Num/Side
    if 'Cabin' in df.columns:
        # Fill NA temporarily for splitting to avoid errors
        cabin = df['Cabin'].fillna('nan/nan/nan')
        df['Deck'] = cabin.apply(lambda x: x.split('/')[0])
        df['CabinNum'] = cabin.apply(lambda x: x.split('/')[1])
        df['Side'] = cabin.apply(lambda x: x.split('/')[2])
        # Replace 'nan' strings back to actual NaN for pipeline handling
        df.loc[df['Deck'] == 'nan', ['Deck', 'CabinNum', 'Side']] = None
        df = df.drop('Cabin', axis=1)

    # Name: Extract Surname
    if 'Name' in df.columns:
        df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notnull(x) else None)
        df = df.drop('Name', axis=1)
        
    return df

# Test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_f = build_features(train)
test_f = build_features(test)

print("Train columns:", train_f.columns.tolist())
print("Train dtypes:\n", train_f.dtypes)
print("\nTest columns:", test_f.columns.tolist())
