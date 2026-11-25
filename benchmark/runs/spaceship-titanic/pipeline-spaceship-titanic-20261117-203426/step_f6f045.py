import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Load
train_df = pd.read_csv('train.csv')

# Build features
train = build_features(train_df.copy())

# Identify columns
target = 'Transported'
# Explicitly identify columns to use, ensuring consistent types
# 'CryoSleep' and 'VIP' are 'object' but contain bool/NaN. Force to string for OHE.
for col in ['CryoSleep', 'VIP']:
    train[col] = train[col].astype(str)

features = [c for c in train.columns if c not in [target, 'PassengerId', 'Name']]
numeric_features = [c for c in features if train[c].dtype in ['int64', 'float64']]
categorical_features = [c for c in features if c not in numeric_features]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Models
models = [
    ("RandomForest", RandomForestClassifier(random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42))
]

# CV
X = train.drop(columns=[target, 'PassengerId'])
# Ensure all categorical columns are string for the pipeline
for col in X.columns:
    if col not in numeric_features:
        X[col] = X[col].astype(str)
        
y = train[target].astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
    # log_experiment is not available here, but I will record the result manually.
