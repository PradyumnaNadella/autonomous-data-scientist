import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
for df in [train, test]:
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    df['Deck'] = df['Cabin'].str.split('/', expand=True)[0]
    df['Side'] = df['Cabin'].str.split('/', expand=True)[2]
    # Total spend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)

features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Deck', 'Side', 'TotalSpend', 'Group']
numeric_features = ['Age', 'TotalSpend']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Models
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

xgb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', XGBClassifier(n_estimators=100, random_state=42))])

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train.drop('Transported', axis=1)
y = train['Transported']

rf_score = cross_val_score(rf, X, y, cv=cv, scoring='accuracy').mean()
xgb_score = cross_val_score(xgb, X, y, cv=cv, scoring='accuracy').mean()

print(f"RF Score: {rf_score}")
print(f"XGB Score: {xgb_score}")
