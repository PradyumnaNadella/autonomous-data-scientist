import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)

# Convert mixed types to string for categorical columns
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Num', 'Side', 'Surname']
for col in cat_cols:
    train[col] = train[col].astype(str)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported'].astype(int)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = [col for col in X.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

def get_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Use lighter models for initial comparison
rf = get_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
xgb = get_pipeline(XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))

rf_score = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()
xgb_score = cross_val_score(xgb, X, y, cv=5, scoring='accuracy').mean()

print(f"RF CV Score: {rf_score}")
print(f"XGB CV Score: {xgb_score}")
