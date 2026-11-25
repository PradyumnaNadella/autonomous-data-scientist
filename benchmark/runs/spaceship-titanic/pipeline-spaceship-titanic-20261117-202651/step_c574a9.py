import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from features import build_features

# Load data
train = pd.read_csv('train.csv')
X_train = build_features(train.drop(columns=['Transported']))
y_train = train['Transported']

# Convert object/bool columns to string to ensure OHE works
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Define final models for experiment log
model_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
model_lgbm_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))])

# Run CV to finalize scores for log_experiment
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score_rf = cross_val_score(model_rf, X_train, y_train, cv=cv, scoring='accuracy').mean()
score_lgbm_tuned = cross_val_score(model_lgbm_tuned, X_train, y_train, cv=cv, scoring='accuracy').mean()

print(f"RF CV: {score_rf}")
print(f"LGBM Tuned CV: {score_lgbm_tuned}")
