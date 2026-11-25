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
test_df = pd.read_csv('test.csv')

# Build features (features.py is expected to return processed df)
# Assuming build_features adds columns and handles basic cleaning
train = build_features(train_df.copy())
test = build_features(test_df.copy())

# Identify columns
target = 'Transported'
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
X = train.drop(columns=[target, 'PassengerId', 'Name'] if 'Name' in train.columns else [target, 'PassengerId'])
y = train[target]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
