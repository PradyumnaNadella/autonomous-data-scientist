import pandas
import numpy as np

def build_features(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    # Handle Cabin split safely
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    return df.drop(columns=['Cabin'])

# Test on files
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')

train_feat = build_features(train)
test_feat = build_features(test)

print("Train columns:", train_feat.columns.tolist())
print("\nTest columns:", test_feat.columns.tolist())

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
