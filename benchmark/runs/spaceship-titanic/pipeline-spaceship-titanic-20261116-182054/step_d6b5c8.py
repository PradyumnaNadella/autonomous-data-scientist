import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load and prepare
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Use build_features
X_raw = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Identify feature types
cat_cols = [c for c in X_raw.columns if X_raw[c].dtype == 'object']
num_cols = [c for c in X_raw.columns if X_raw[c].dtype in ['int64', 'float64']]

# Define preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# 2. Setup Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 3. CV loop
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipeline = Pipeline([('pre', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X_raw, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {results[name]:.4f}")
