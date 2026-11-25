import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering: extract Deck, Side from Cabin
def extract_cabin(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['Cabin', 'Name', 'Num'])
    return df

train = extract_cabin(train)
test = extract_cabin(test)

X = train.drop(columns=['Transported'])
y = train['Transported']

# Identify column types
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ]
)

# Model 1: Random Forest
rf_pipe = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
scores_rf = cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy')
print(f"RF CV Score: {scores_rf.mean():.4f}")

# Model 2: LGBM
lgbm_pipe = Pipeline([('pre', preprocessor), ('clf', LGBMClassifier(random_state=42, verbose=-1))])
scores_lgbm = cross_val_score(lgbm_pipe, X, y, cv=5, scoring='accuracy')
print(f"LGBM CV Score: {scores_lgbm.mean():.4f}")
