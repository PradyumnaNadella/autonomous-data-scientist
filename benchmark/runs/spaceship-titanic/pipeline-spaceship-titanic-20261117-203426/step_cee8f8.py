import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from features import build_features

# Load and prepare
train_df = build_features(pd.read_csv('train.csv'))
X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported'].astype(int)

# Clean: Convert bools to strings explicitly BEFORE pipeline
for col in X.columns:
    if X[col].dtype == 'bool' or X[col].dtype == 'object':
        X[col] = X[col].astype(str)

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# Run CV
for name, model in [("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
                   ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42))]:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
    # log_experiment for each
    # This must be done via the tool call outside this script, I'll log them in the response.
