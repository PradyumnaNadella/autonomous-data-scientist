import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PassengerId: Group is the part before '_'
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    
    # Cabin: Split into Deck, Num, Side
    # handle cases where Cabin is NaN
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1] # Keep as object for now, Imputer handles
    df['Side'] = cabin_split[2]
    
    # Drop Name as per plan (or keep as a pass-through if needed, but per plan it's dropped)
    # Keeping PassengerId and Transported as required
    
    # Drop original columns that were split
    df = df.drop(columns=['Cabin', 'Name'])
    
    return df

# Test on train/test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train dtypes:\n", train_feat.dtypes)
print("\nTest dtypes:\n", test_feat.dtypes)
print("\nTrain columns:", train_feat.columns.tolist())

# Save to features.py
code = """
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    return df.drop(columns=['Cabin', 'Name'])
"""
open('features.py', 'w').write(code)
