import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)

# Select features
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Model 1: Random Forest
rf_pipe = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score_rf = cross_val_score(rf_pipe, X, y, cv=cv, scoring='accuracy').mean()
print(f"RF CV Score: {score_rf}")

# Model 2: Logistic Regression
lr_pipe = Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
score_lr = cross_val_score(lr_pipe, X, y, cv=cv, scoring='accuracy').mean()
print(f"LR CV Score: {score_lr}")

# Log experiments
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Just to make it clean
import os
import sys

# Re-importing because this is a new process
from sklearn.utils import all_estimators
# Skipping manual logging, using the provided function
# Note: I have to call the tool log_experiment directly.

print("Logging experiments...")
