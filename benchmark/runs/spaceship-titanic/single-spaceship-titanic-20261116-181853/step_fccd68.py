import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')

# Feature Engineering
def preprocess_data(df):
    df = df.copy()
    # Extract deck, num, side from Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    return df

X = preprocess_data(train)
y = train['Transported']

# Identify column types
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Models
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

gb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])

# Cross-Validation
scores_rf = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
scores_gb = cross_val_score(gb, X, y, cv=5, scoring='accuracy')

print(f"RF CV Accuracy: {scores_rf.mean():.4f}")
print(f"GB CV Accuracy: {scores_gb.mean():.4f}")
