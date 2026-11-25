import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import json

# Load and process
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features using the provided script
df = build_features(train.drop('Transported', axis=1))
df['Transported'] = train['Transported']

# Identify features (Drop ID/Names/High Cardinality Raw)
# Based on instructions, we drop PassengerId and Name. 
# We'll use: HomePlanet, CryoSleep, Cabin, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
# Cabin is split into Deck, Side (Num is numerical but high cardinality)
# Let's see what build_features did.
print(df.columns)

# Define features for the model
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Models to compare
models = {
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, df.drop('Transported', axis=1), df['Transported'], cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
