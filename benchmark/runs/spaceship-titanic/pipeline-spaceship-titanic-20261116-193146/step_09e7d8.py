from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

# Identify column types
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Num', 'Side']

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

def log_experiment(name, model, X, y):
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")

log_experiment("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42), X, y)
log_experiment("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42), X, y)
