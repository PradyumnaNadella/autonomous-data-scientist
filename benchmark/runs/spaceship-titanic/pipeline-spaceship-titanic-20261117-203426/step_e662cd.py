import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Load
train_df = pd.read_csv('train.csv')
train = build_features(train_df.copy())
target = 'Transported'

# CRITICAL FIX: Convert all columns intended for OneHotEncoder to string.
# The error "TypeError: '<' not supported between instances of 'str' and 'bool'"
# implies that the DataFrame passed to OneHotEncoder still has columns with mixed bool/str.
# We must cast ALL columns to string if we want OHE to handle them without error.

# Drop IDs
if 'Name' in train.columns:
    train = train.drop(columns=['Name'])
train = train.drop(columns=['PassengerId'])

# Convert EVERYTHING except numeric to string
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in train.columns:
    if col != target and col not in numeric_cols:
        train[col] = train[col].astype(str)

# Separate
X = train.drop(columns=[target])
y = train[target].astype(int)

categorical_features = [c for c in X.columns if c not in numeric_cols]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_features)])

# Model
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"HistGradientBoosting: {scores.mean():.4f}")
