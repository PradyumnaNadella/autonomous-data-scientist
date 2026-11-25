import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json

def preprocess(df):
    df = df.copy()
    # Cabin splitting
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    # Spending
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    return df.drop(['Name', 'Cabin', 'PassengerId'], axis=1)

train = pd.read_csv('train.csv')
train = preprocess(train)

X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)

cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'VIP']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'IsSpending']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

results = []
models = [
    ('RF', RandomForestClassifier(random_state=42, n_estimators=100)),
    ('LGBM', LGBMClassifier(random_state=42, n_estimators=100, verbose=-1))
]

for name, model in models:
    clf = Pipeline([('pre', preprocessor), ('clf', model)])
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    results.append({'model': name, 'score': scores.mean()})
    print(f"{name}: {scores.mean()}")

with open('experiments.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
