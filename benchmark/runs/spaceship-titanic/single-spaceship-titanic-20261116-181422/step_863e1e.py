import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported']

def preprocess(df):
    df = df.copy()
    # Convert all object columns to string to avoid mixed types
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
    
    # Extract deck, side from Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
    return df

X = preprocess(X)
# Target is bool, let's keep it as is, but model handles it.

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

# Model 1: Random Forest
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Model 2: XGBoost
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
])

# CV
scores_rf = cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy')
scores_xgb = cross_val_score(xgb_pipe, X, y, cv=5, scoring='accuracy')

print(f"RF CV Score: {scores_rf.mean():.4f}")
print(f"XGB CV Score: {scores_xgb.mean():.4f}")
