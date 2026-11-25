from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from features import build_features

# Load and prepare
train = build_features(pd.read_csv('train.csv'))
X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

# Define transformers
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group'] # Group is high cardinality but we just treat it as cat here for now
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Models
models = [
    ('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('XGB', XGBClassifier(n_estimators=100, random_state=42))
]

# Evaluate
results = {}
for name, model in models:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name} CV Accuracy: {results[name]:.4f}")
