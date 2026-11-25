import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np

# Load
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Feature Engineering
X_train = build_features(df_train.drop('Transported', axis=1))
y_train = df_train['Transported'].astype(int)
X_test = build_features(df_test)

# Identify feature types
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X_train.select_dtypes(include=['number']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_features),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    clf = Pipeline([('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
