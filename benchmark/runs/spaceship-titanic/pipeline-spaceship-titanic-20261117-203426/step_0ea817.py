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
# Drop problematic columns
drop_cols = [target, 'PassengerId', 'Name']
if 'Name' in train.columns:
    train = train.drop(columns=['Name'])
    
# Identify features
X = train.drop(columns=[target, 'PassengerId'])
y = train[target].astype(int)

# Separate types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Note: OneHotEncoder by default returns sparse matrix. HistGradientBoostingClassifier requires dense.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Models
models = [
    ("RandomForest", RandomForestClassifier(random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42))
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
