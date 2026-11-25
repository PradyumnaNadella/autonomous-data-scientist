import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Re-run setup
df_train = pd.read_csv('train.csv')
X_train = build_features(df_train.drop('Transported', axis=1))
y_train = df_train['Transported'].astype(int)

for col in X_train.columns:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(str)
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str)

cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
num_features = X_train.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_features),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

# Experiments
rf_clf = Pipeline([('preprocessor', preprocessor), 
                   ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
rf_score = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"RandomForest: {rf_score:.4f}")

xgb_default_clf = Pipeline([('preprocessor', preprocessor), 
                           ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
xgb_default_score = cross_val_score(xgb_default_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"XGBoost (default): {xgb_default_score:.4f}")

xgb_tuned_clf = Pipeline([('preprocessor', preprocessor), 
                          ('model', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42))])
xgb_tuned_score = cross_val_score(xgb_tuned_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"XGBoost (tuned): {xgb_tuned_score:.4f}")
