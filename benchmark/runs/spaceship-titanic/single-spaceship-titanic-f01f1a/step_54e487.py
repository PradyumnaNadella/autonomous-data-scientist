import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')

# Feature Engineering: Extract Group and Deck from Cabin
def preprocess_features(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notnull(x) else np.nan)
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notnull(x) else np.nan)
    df = df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
    return df

X = preprocess_features(train.drop('Transported', axis=1))
y = train['Transported']

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Model 1: Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_score = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='accuracy').mean()

# Model 2: Logistic Regression
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

lr_score = cross_val_score(lr_pipeline, X, y, cv=cv, scoring='accuracy').mean()

print(f"RF CV Score: {rf_score}")
print(f"LR CV Score: {lr_score}")

from sklearn.base import BaseEstimator
# Log experiments
import json
# Using tool call directly
