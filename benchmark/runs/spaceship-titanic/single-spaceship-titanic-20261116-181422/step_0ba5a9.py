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
X = train.drop('Transported', axis=1)
y = train['Transported']

# Feature Engineering
def preprocess(df):
    df = df.copy()
    # Extract deck, num, side from Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
    # Fill numerical missing with median
    # Fill categorical with mode or 'Unknown'
    return df

X = preprocess(X)

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
    ('classifier', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# CV
scores_rf = cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy')
scores_xgb = cross_val_score(xgb_pipe, X, y, cv=5, scoring='accuracy')

print(f"RF CV Score: {scores_rf.mean():.4f}")
print(f"XGB CV Score: {scores_xgb.mean():.4f}")
