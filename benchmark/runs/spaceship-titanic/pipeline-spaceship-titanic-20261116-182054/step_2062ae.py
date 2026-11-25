import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from features import build_features

# Load and prepare
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported']
X_test = build_features(test_df)

for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

drop_cols = ['PassengerId', 'Name', 'Cabin']
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
passenger_ids = test_df['PassengerId']
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

# Chosen Model
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
output = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': preds})
output.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")

# Feature Importance (Coefficients)
ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(ohe_feature_names)
coefs = model.named_steps['classifier'].coef_[0]
importance = pd.Series(coefs, index=all_feature_names).abs().sort_values(ascending=False)
print("Top 5 features:")
print(importance.head(5))
