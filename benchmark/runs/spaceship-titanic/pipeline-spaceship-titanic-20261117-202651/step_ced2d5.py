import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from features import build_features

def get_pipeline(model):
    train = build_features(pd.read_csv('train.csv'))
    # Ensure all categorical columns are string type to avoid TypeError in OHE
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
    for col in cat_cols:
        train[col] = train[col].astype(str)

    X = train.drop(columns=['Transported', 'PassengerId'])
    y = train['Transported']
    
    num_cols = X.select_dtypes(include=['float64']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])
    
    return Pipeline([('preprocessor', preprocessor), ('model', model)])

# Evaluation
train = build_features(pd.read_csv('train.csv'))
# Fix types again
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
for col in cat_cols:
    train[col] = train[col].astype(str)
X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

rf_pipe = get_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
lgbm_pipe = get_pipeline(LGBMClassifier(random_state=42, verbose=-1))

rf_scores = cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy')
lgbm_scores = cross_val_score(lgbm_pipe, X, y, cv=5, scoring='accuracy')

print(f"Random Forest CV Accuracy: {rf_scores.mean():.4f}")
print(f"LGBM CV Accuracy: {lgbm_scores.mean():.4f}")
