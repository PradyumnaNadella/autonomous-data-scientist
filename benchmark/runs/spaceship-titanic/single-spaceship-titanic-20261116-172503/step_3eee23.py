import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported']

# Feature Engineering
def preprocess(df):
    df = df.copy()
    # Extract Group and Member info from PassengerId
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    # Extract Deck, Num, Side from Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    # Simple feature for total spend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    return df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

X_processed = preprocess(X)

# Define column types
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model 1: Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Model 2: Gradient Boosting
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GradientBoostingClassifier(random_state=42))])

# CV
scores_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='accuracy')
scores_gb = cross_val_score(gb_pipeline, X, y, cv=5, scoring='accuracy')

print(f"RF CV Score: {scores_rf.mean():.4f}")
print(f"GB CV Score: {scores_gb.mean():.4f}")
