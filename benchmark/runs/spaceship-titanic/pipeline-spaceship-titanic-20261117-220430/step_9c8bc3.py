import pandas as pd

def build_features(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.copy()
    
    # PassengerId: Extract Group
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    # Cabin: Split into Deck, Num, Side
    # Use fillna to handle missing values before splitting, though row-wise split is safe
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    
    # Drop raw Cabin as it's split
    df = df.drop(columns=['Cabin'])
    
    return df

# Test on files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train columns:", train_feat.columns.tolist())
print("Train dtypes:\n", train_feat.dtypes)
print("\nTest columns:", test_feat.columns.tolist())
print("Test dtypes:\n", test_feat.dtypes)

# Save the function to a file
with open('features.py', 'w') as f:
    f.write('''
import pandas
def build_features(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    return df.drop(columns=['Cabin'])
''')
