import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from features import build_features

train = build_features(pd.read_csv('train.csv'))

# Convert all potential problematic columns (bool/mixed) to string for OneHotEncoder
for col in ['CryoSleep', 'VIP']:
    train[col] = train[col].astype(str)

# Ensure 'Num' is numeric and handle missing values there too, but let's cast it to float
train['Num'] = pd.to_numeric(train['Num'], errors='coerce')

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

models = [
    ('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('XGB', XGBClassifier(n_estimators=100, random_state=42))
]

for name, model in models:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
