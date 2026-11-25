import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from features import build_features

# Load Data
train = pd.read_csv('train.csv')
train = build_features(train)

# Separate Target
X = train.drop(columns=['Transported', 'PassengerId', 'Name', 'Group']) # Excluding High Cardinality
y = train['Transported']

# Pipeline
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Models
rf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
gb = Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])

# Evaluate
for name, model in [('RandomForest', rf), ('GradientBoosting', gb)]:
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
