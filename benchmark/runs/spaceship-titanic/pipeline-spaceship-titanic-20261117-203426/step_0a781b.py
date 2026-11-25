import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from features import build_features

# Load and prepare
train_df = build_features(pd.read_csv('train.csv'))
X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported'].astype(int)

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Define Preprocessor
# Simple preprocessing: convert all categorical columns to string
for col in categorical_features:
    X[col] = X[col].astype(str)

# Preprocessor for both
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ])

# Run CV for HistGradientBoosting
pipeline_hgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(categorical_features=[False]*len(numeric_features) + [True]*len(categorical_features), random_state=42))])
scores_hgb = cross_val_score(pipeline_hgb, X, y, cv=5, scoring='accuracy')
print(f"HistGradientBoosting CV Accuracy: {scores_hgb.mean():.4f}")

# And for RF, OHE is needed
preprocessor_rf = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor_rf), ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))])
scores_rf = cross_val_score(pipeline_rf, X, y, cv=5, scoring='accuracy')
print(f"RandomForest CV Accuracy: {scores_rf.mean():.4f}")
