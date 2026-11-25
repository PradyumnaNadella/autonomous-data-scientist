import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Simplified feature build
def build_features(df):
    df = df.copy()
    # Cabin parsing
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Side'] = cabin[2]
    # Billing
    cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[cols] = df[cols].fillna(0)
    df['TotalSpend'] = df[cols].sum(axis=1)
    df = df.drop(columns=['Name', 'Cabin', 'PassengerId'])
    return df

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y = train['Transported'].astype(int)
X = build_features(train.drop(columns=['Transported']))
X_test = build_features(test)

num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# CV
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RF
pipe_rf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
scores_rf = cross_val_score(pipe_rf, X, y, cv=cv, scoring='accuracy')
print(f"RF Mean CV: {scores_rf.mean():.4f}")

# XGB
pipe_xgb = Pipeline([('pre', preprocessor), ('clf', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])
scores_xgb = cross_val_score(pipe_xgb, X, y, cv=cv, scoring='accuracy')
print(f"XGB Mean CV: {scores_xgb.mean():.4f}")
