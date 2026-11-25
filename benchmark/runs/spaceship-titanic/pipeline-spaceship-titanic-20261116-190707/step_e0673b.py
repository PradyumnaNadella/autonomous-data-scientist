import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features import build_features
import json

# Load data
train = pd.read_csv('train.csv')
y = train['Transported'].astype(int)
X = build_features(train).drop(['Transported', 'PassengerId'], axis=1)

# Define column groups
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']

# Force types
X[num_cols] = X[num_cols].apply(pd.to_numeric, errors='coerce')
X[cat_cols] = X[cat_cols].astype(str)

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Tune LightGBM
pipe = Pipeline([('pre', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])
param_distributions = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [5, 10, -1]
}

search = RandomizedSearchCV(pipe, param_distributions, cv=5, scoring='accuracy', n_iter=5, n_jobs=-1, random_state=42)
search.fit(X, y)
print(f"Best score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# Log experiment
def log_experiment(model_name, score, notes):
    with open('experiments.jsonl', 'a') as f:
        f.write(json.dumps({'model': model_name, 'cv_score': score, 'notes': notes}) + '\n')

log_experiment("LightGBM Tuned", search.best_score_, f"Params: {search.best_params_}")

# Feature Importance
model = search.best_estimator_.named_steps['model']
pre = search.best_estimator_.named_steps['pre']
cat_features = pre.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
all_features = num_cols + list(cat_features)
importances = model.feature_importances_
top_5_idx = np.argsort(importances)[-5:][::-1]
print("Top 5 features:")
for i in top_5_idx:
    print(f"{all_features[i]}: {importances[i]}")

# Save model
import joblib
joblib.dump(search.best_estimator_, 'model.joblib')
