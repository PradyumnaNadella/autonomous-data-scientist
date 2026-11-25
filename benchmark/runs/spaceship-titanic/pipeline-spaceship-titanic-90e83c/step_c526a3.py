import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json

def preprocess(df):
    df = df.copy()
    # Cabin splitting
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    # PassengerID group
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    # Name surname
    df['Surname'] = df['Name'].str.split(' ', expand=True)[1]
    # Spending
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    return df.drop(['Name', 'Cabin'], axis=1)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = preprocess(train)
test = preprocess(test)

X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)

cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Group', 'Surname']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'IsSpending']

# Drop columns that have too many unique values for simple OHE if needed, 
# but let's stick to the plan for now.
# Dropping 'Group' and 'Surname' for this quick test as they might be high cardinality.
features_to_use = [c for c in X.columns if c not in ['Group', 'Surname', 'PassengerId']]
cat_cols = [c for c in cat_cols if c in features_to_use]
num_cols = [c for c in num_cols if c in features_to_use]

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Experiments
results = []
models = [
    ('RF', RandomForestClassifier(random_state=42)),
    ('LGBM', LGBMClassifier(random_state=42, verbose=-1))
]

for name, model in models:
    clf = Pipeline([('pre', preprocessor), ('clf', model)])
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    results.append({'model': name, 'score': scores.mean()})
    print(f"{name}: {scores.mean()}")

with open('experiments.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
