import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Load
train = pd.read_csv('train.csv')
train = build_features(train)

# Fix Issue 1 & 3: Drop Group and PassengerId as they are high cardinality or IDs
X = train.drop(columns=['Transported', 'PassengerId', 'Group'])
y = train['Transported'].astype(int)

# Identify features
cat_cols = ['Deck', 'Side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ]
)

# Compare models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model, name in [
    (RandomForestClassifier(random_state=42), "Random Forest"),
    (GradientBoostingClassifier(random_state=42), "Gradient Boosting")
]:
    clf = Pipeline([('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
