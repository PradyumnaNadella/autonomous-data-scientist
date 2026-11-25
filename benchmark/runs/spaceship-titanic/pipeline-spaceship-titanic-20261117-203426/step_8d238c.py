import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import numpy as np

# Load data
train_df = pd.read_csv('train.csv')

# Feature Engineering
train_feat = build_features(train_df)

# Drop non-feature columns
drop_cols = ['PassengerId', 'Name']
train_feat = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])

X = train_feat.drop(columns=['Transported'])
y = train_feat['Transported']

# Convert all categorical to string to avoid bool/str mix
# In ColumnTransformer, X is a DataFrame when passed to sub-pipelines.
# Wait, maybe it's numpy because it came out of another transformer?
def cast_to_string(X):
    return pd.DataFrame(X).astype(str)

categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Pipeline
# Note: Input to preprocessor is X (DataFrame), so categoricals are fine here.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('cast', FunctionTransformer(cast_to_string)),
                          ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

def get_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# CV (3 folds to save time)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Models
models = {
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=50),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    clf = get_pipeline(model)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV Score: {scores.mean():.4f}")
