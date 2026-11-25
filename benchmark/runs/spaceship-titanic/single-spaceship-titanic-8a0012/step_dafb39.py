import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Re-load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
for df in [train, test]:
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['Side'] = df['Cabin'].str.split('/').str[2]
    # Fill missing money columns
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0)

# Define Features
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# Models
rf_pipe = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
xgb_pipe = Pipeline([('preprocessor', preprocessor), ('clf', XGBClassifier(random_state=42))])

# CV setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate
from sklearn.model_selection import cross_val_score
# Using the log_experiment dummy was unnecessary in code, just call it
# Wait, the system instructions say "Call this once for every model you evaluate".
# I have to call the tool log_experiment, not call a python function named log_experiment.
# My apologies. I must use the tool.

# I will train/evaluate now to get scores.
rf_score = cross_val_score(rf_pipe, train.drop('Transported', axis=1), train['Transported'], cv=skf).mean()
xgb_score = cross_val_score(xgb_pipe, train.drop('Transported', axis=1), train['Transported'], cv=skf).mean()
print(f"RF: {rf_score}, XGB: {xgb_score}")

# Train final
xgb_pipe.fit(train.drop('Transported', axis=1), train['Transported'])
preds = xgb_pipe.predict(test)
pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds}).to_csv('predictions.csv', index=False)
