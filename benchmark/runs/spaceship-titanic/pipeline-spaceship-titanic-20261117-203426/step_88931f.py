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

# Drop IDs and Names immediately
if 'Name' in train.columns:
    train = train.drop(columns=['Name'])

# Identify numeric cols specifically to avoid confusion
# Original columns to treat as numeric
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Separate
X = train.drop(columns=[target, 'PassengerId'])
y = train[target].astype(int)

categorical_features = [c for c in X.columns if c not in numeric_cols]

# Simplified Preprocessing
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
