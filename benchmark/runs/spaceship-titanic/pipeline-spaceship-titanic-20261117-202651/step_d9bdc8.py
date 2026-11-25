import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

# Fit model
model_lgbm_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))])
model_lgbm_tuned.fit(X_train, y_train)

# Feature importance
importances = model_lgbm_tuned.named_steps['classifier'].feature_importances_
feature_names = numeric_features + list(model_lgbm_tuned.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_features))

feat_imp = pd.Series(importances, index=feature_names)
print(feat_imp.sort_values(ascending=False).head(5))
