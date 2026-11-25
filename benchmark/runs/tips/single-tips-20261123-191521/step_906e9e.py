import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import numpy as np

train = pd.read_csv('train.csv')
X = train.drop(['row_id', 'tip'], axis=1)
y = train['tip']

numeric_features = ['total_bill', 'size']
categorical_features = ['sex', 'smoker', 'day', 'time']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model 1: Ridge
ridge_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

# Model 2: RandomForest
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

ridge_scores = cross_val_score(ridge_pipe, X, y, cv=cv, scoring='neg_mean_absolute_error')
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring='neg_mean_absolute_error')

print(f"Ridge MAE: {-ridge_scores.mean()}")
print(f"RF MAE: {-rf_scores.mean()}")
