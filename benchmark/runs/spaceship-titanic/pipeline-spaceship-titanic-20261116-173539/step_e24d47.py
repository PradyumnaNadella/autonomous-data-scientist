import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

# Updated features.py logic
def build_features_fixed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = pd.to_numeric(cabin_split[1])
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Name', 'Cabin'])
    return df

df = pd.read_csv('train.csv')
df = build_features_fixed(df)

target = 'Transported'
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'Group']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']

# Correcting types for ColumnTransformer
for col in cat_features:
    df[col] = df[col].astype(str)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# 1. Random Forest
rf_pipe = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
scores_rf = cross_val_score(rf_pipe, df.drop(columns=[target, 'PassengerId']), df[target], cv=5, scoring='accuracy')
print(f"RF CV: {np.mean(scores_rf):.4f}")

# 2. XGBoost
xgb_pipe = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
scores_xgb = cross_val_score(xgb_pipe, df.drop(columns=[target, 'PassengerId']), df[target], cv=5, scoring='accuracy')
print(f"XGB CV: {np.mean(scores_xgb):.4f}")
