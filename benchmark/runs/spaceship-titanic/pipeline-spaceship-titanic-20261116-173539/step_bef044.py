import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from features import build_features

# Load and process
train = pd.read_csv('train.csv')
train = build_features(train)

# Convert all potential categorical columns to string to avoid mixed types
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']:
    train[col] = train[col].astype(str)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported'].astype(str) # ensure target is consistent

# Ensure 'Num' is numeric
train['Num'] = pd.to_numeric(train['Num'], errors='coerce')

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']

# Build Preprocessor
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# Define models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
