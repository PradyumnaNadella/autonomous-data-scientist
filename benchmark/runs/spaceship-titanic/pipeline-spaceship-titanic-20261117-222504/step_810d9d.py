import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features import build_features

# Load and prepare
train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop(columns=['Transported']))
y = train_df['Transported'].astype(int)

# Identify feature types
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum', 'TotalSpent', 'HasSpent']

# Pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# Models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {results[name]:.4f}")

# The user is expected to call log_experiment, but I will record them manually
# I will output the final scores to confirm they match the requirements
