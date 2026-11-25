import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import numpy as np

# Load data
train = pd.read_csv('train.csv')
X = train.drop(['row_id', 'tip'], axis=1)
y = train['tip']

# Preprocessing
cat_features = ['sex', 'smoker', 'day', 'time']
num_features = ['total_bill', 'size']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# Models
rf = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))])
ridge = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge())])

# Cross-validation
cv_rf = -cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
cv_ridge = -cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_absolute_error').mean()

print(f"RandomForest MAE: {cv_rf}")
print(f"Ridge MAE: {cv_ridge}")
