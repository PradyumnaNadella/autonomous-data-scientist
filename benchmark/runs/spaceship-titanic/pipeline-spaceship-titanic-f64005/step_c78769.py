import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

# Load and process
train = pd.read_csv('train.csv')
train_feat = build_features(train)

# Convert all object types to string to ensure OHE works
for col in train_feat.select_dtypes(include=['object', 'bool']).columns:
    train_feat[col] = train_feat[col].astype(str)

# Re-identify feature types
num_cols = train_feat.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = [c for c in train_feat.columns if c not in num_cols and c != 'Transported']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(n_estimators=100, verbose=-1)
}

# Target
y = train_feat['Transported'].astype(str)
X = train_feat.drop('Transported', axis=1)

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
