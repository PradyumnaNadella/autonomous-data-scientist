import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')

def build_features(df):
    df = df.copy()
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    return df

X = build_features(train.drop(columns=['Transported']))
y = train['Transported'].astype(int)

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

clf = Pipeline([('prep', preprocessor), ('model', RandomForestClassifier(n_estimators=50, random_state=42))])
scores = cross_val_score(clf, X, y, cv=3)
print(f"RF Score: {scores.mean()}")
