import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
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

# Define Preprocessor: Use sparse matrix to speed up
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]), categorical_features)
    ])

# Define models
models = {
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42)
}

# The problem in the last attempt was the FunctionTransformer for strings. 
# HistGradientBoostingClassifier handles categorical data natively, so I don't necessarily need OHE.
# Let's try skipping OHE for HistGradientBoosting, but I need it for RandomForest.
# Actually, I can just convert the categorical columns to categorical dtype before pipeline.

for col in categorical_features:
    X[col] = X[col].astype(str).astype('category')

# Re-identify numeric/categorical after type conversion
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['category']).columns.tolist()

# For HistGradientBoosting, use OrdinalEncoder or handle natively
from sklearn.preprocessing import OrdinalEncoder
preprocessor_hgb = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ])

# Run CV for HistGradientBoosting
pipeline_hgb = Pipeline(steps=[('preprocessor', preprocessor_hgb), ('classifier', HistGradientBoostingClassifier(categorical_features=[True]*len(categorical_features), random_state=42))])
scores_hgb = cross_val_score(pipeline_hgb, X, y, cv=5, scoring='accuracy')
print(f"HistGradientBoosting CV Accuracy: {scores_hgb.mean():.4f}")

# And for RF, OHE is needed
preprocessor_rf = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor_rf), ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))])
scores_rf = cross_val_score(pipeline_rf, X, y, cv=5, scoring='accuracy')
print(f"RandomForest CV Accuracy: {scores_rf.mean():.4f}")
