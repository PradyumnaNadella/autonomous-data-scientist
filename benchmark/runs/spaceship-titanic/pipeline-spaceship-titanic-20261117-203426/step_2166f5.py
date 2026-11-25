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
train = build_features(train_df.copy())

# Target/Features setup
target = 'Transported'
if 'Name' in train.columns:
    train = train.drop(columns=['Name'])

# Fix mixed types: Convert to string, but handle nan specifically
for col in train.columns:
    if col != target and col != 'PassengerId':
        train[col] = train[col].astype(str).replace('nan', 'missing')

X = train.drop(columns=[target, 'PassengerId'])
y = train[target].astype(int)

# Separate types - need to re-identify after string conversion
numeric_features = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
categorical_features = [c for c in X.columns if c not in numeric_features]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Models
models = [
    ("RandomForest", RandomForestClassifier(random_state=42, n_estimators=50)),
    ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42))
]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
