import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
target = 'Transported'
X = train.drop(columns=[target])
y = train[target].astype(int)

# Feature engineering: extract deck, num, side from Cabin
def preprocess(df):
    df = df.copy()
    if 'Cabin' in df.columns:
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df = df.drop(columns=['Cabin'])
    # Drop Name and PassengerId (Name is too high cardinality, PassengerId is not informative)
    df = df.drop(columns=['Name', 'PassengerId'], errors='ignore')
    return df

X = preprocess(X)

# Define columns
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Pipelines
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

# Model 1: Random Forest
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
rf_score = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy').mean()
print(f"Random Forest CV Score: {rf_score}")

# Model 2: XGBoost
xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', XGBClassifier(n_estimators=100, random_state=42))])
xgb_score = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy').mean()
print(f"XGBoost CV Score: {xgb_score}")
