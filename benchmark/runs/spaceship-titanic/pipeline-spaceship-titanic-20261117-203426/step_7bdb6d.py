import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Load data
train_df = pd.read_csv('train.csv')

# Feature Engineering
train_df = build_features(train_df)

# Drop irrelevant
drop_cols = ['PassengerId', 'Name', 'Cabin']
X = train_df.drop(columns=drop_cols + ['Transported'])
y = train_df['Transported'].astype(int)

# Cast everything categorical to string to avoid mixed types
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']:
    X[col] = X[col].astype(str)

# Cast numeric to float
for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']:
    X[col] = pd.to_numeric(X[col], errors='coerce')

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']

# Preprocessing Pipeline
# Added sparse_output=False for dense matrix
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_features)
])

# Compare Models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'HistGBM': HistGradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f'{name} CV Accuracy: {scores.mean():.4f}')
