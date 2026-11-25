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
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['Side'] = df['Cabin'].str.split('/').str[2]
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    # Fill missing money columns
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0)

# Define Features
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# CV setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
rf_pipe = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
xgb_pipe = Pipeline([('preprocessor', preprocessor), ('clf', XGBClassifier(random_state=42))])

# Evaluate
from sklearn.model_selection import cross_val_score
rf_score = cross_val_score(rf_pipe, train.drop('Transported', axis=1), train['Transported'], cv=skf).mean()
xgb_score = cross_val_score(xgb_pipe, train.drop('Transported', axis=1), train['Transported'], cv=skf).mean()

print(f"RF Score: {rf_score}")
print(f"XGB Score: {xgb_score}")
