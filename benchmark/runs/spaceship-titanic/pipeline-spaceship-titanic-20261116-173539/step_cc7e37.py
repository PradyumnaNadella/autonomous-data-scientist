import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']:
    train[col] = train[col].astype(str)
train['Num'] = pd.to_numeric(train['Num'], errors='coerce')
X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported'].astype(bool)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss'))
])
model.fit(X, y)

# Get feature names
cat_ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
cat_feature_names = cat_ohe.get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

# Get importances
importances = model.named_steps['classifier'].feature_importances_
feat_importances = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
print(feat_importances.head(5))
