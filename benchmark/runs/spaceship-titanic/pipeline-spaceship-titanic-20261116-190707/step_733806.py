import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import build_features

train_df = pd.read_csv('train.csv')
X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported'].astype(int)

drop_cols = ['PassengerId', 'Surname', 'Group', 'Num']
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(str).replace('nan', 'missing')
X_train.drop(columns=drop_cols, inplace=True)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('classifier', XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss'))])
model.fit(X_train, y_train)

# Feature Importance
feature_names = numeric_features.tolist() + \
                model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features).tolist()

importances = model.named_steps['classifier'].feature_importances_
feature_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
print(feature_imp_df.sort_values(by='importance', ascending=False).head(5))
