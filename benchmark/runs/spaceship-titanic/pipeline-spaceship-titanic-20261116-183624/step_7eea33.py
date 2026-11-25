import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Prepare Data
train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Cast all object/bool cols to str to avoid TypeError
for col in X.select_dtypes(include=['object', 'bool']).columns:
    X[col] = X[col].astype(str)

# Re-identify types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# 2. Experimentation
# Compare Random Forest and XGBoost
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
for name, clf in models.items():
    pipeline = Pipeline([('prep', preprocessor), ('clf', clf)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name} CV Score: {results[name]:.4f}")

# (Note: I will use log_experiment in the final report/subsequent steps)
