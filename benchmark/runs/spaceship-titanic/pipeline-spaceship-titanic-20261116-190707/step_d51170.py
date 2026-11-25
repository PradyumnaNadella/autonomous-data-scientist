import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from features import build_features
import joblib

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocess
X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported'].astype(int)
X_test = build_features(test_df)

# Define pipeline
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

def get_model(clf):
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomForest
rf_pipe = get_model(RandomForestClassifier(random_state=42))
rf_score = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring='accuracy').mean()

# XGBoost
xgb_pipe = get_model(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
xgb_score = cross_val_score(xgb_pipe, X_train, y_train, cv=cv, scoring='accuracy').mean()

print(f"RF CV: {rf_score}")
print(f"XGB CV: {xgb_score}")
