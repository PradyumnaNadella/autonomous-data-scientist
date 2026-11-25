import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from features import build_features
import joblib

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features
X_train = build_features(train.drop(columns=['Transported']))
y_train = train['Transported']
X_test = build_features(test)

# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Pipeline
model_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
model_lgbm = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LGBMClassifier(random_state=42, verbose=-1))])

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score_rf = cross_val_score(model_rf, X_train, y_train, cv=cv, scoring='accuracy').mean()
score_lgbm = cross_val_score(model_lgbm, X_train, y_train, cv=cv, scoring='accuracy').mean()

print(f"RF CV: {score_rf}")
print(f"LGBM CV: {score_lgbm}")
