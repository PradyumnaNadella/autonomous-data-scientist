import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from features import build_features

# 1. Load and prepare
df = pd.read_csv('train.csv')
X = build_features(df.drop('Transported', axis=1))
y = df['Transported']

# Ensure object columns are strings
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype(str)

to_drop = ['PassengerId', 'Name', 'Cabin'] 
X = X.drop(columns=[c for c in to_drop if c in X.columns])

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# RF Tuning
clf_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

param_dist = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

search = RandomizedSearchCV(clf_rf, param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
search.fit(X, y)

print(f"Best RF Score: {search.best_score_}")
print(f"Best Params: {search.best_params_}")
