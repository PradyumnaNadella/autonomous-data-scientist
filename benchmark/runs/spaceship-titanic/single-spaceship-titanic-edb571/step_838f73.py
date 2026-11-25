import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering: Extract deck, num, side from Cabin
def preprocess(df):
    df = df.copy()
    # Handle Cabin split
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    return df

X = preprocess(train.drop(columns=['Transported']))
y = train['Transported'].astype(int)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Model 1: Random Forest
rf_model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
scores_rf = cross_val_score(rf_model, X, y, cv=5)
print(f"RF CV Score: {scores_rf.mean()}")

# Model 2: Logistic Regression
lr_model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
scores_lr = cross_val_score(lr_model, X, y, cv=5)
print(f"LR CV Score: {scores_lr.mean()}")

import joblib
joblib.dump(rf_model, 'best_model.joblib')
