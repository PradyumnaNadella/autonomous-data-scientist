# Re-try without external dependency suspicion (simple pandas/sklearn)
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load data locally in this process
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features (inline to be safe)
def build_features(df):
    df = df.copy()
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Cabin', 'Name'])
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    return df.drop(columns=['PassengerId'])

X = build_features(train.drop(columns=['Transported']))
y = train['Transported'].astype(int)

# Column Transformer
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'IsSpending']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# CV Function
def evaluate(model_cls, **kwargs):
    clf = Pipeline([('prep', preprocessor), ('model', model_cls(**kwargs))])
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# Models
rf_score = evaluate(RandomForestClassifier, n_estimators=100, random_state=42)
xgb_score = evaluate(XGBClassifier, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
lgbm_score = evaluate(LGBMClassifier, n_estimators=100, random_state=42)

print(f"RF: {rf_score}")
print(f"XGB: {xgb_score}")
print(f"LGBM: {lgbm_score}")
