import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')

# Convert boolean columns to strings before processing
def feature_engineering(df):
    df = df.copy()
    # Convert bools to str
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
             df[col] = df[col].astype(str)
             
    # Extract deck, num, side from Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(columns=['Cabin', 'Name', 'PassengerId'], inplace=True)
    return df

X = feature_engineering(train.drop(columns=['Transported']))
y = train['Transported']

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Ensure numeric columns are actually numeric (imputer will handle NaNs)
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Model 1: RandomForest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Model 2: GradientBoosting
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GradientBoostingClassifier(random_state=42))])

# Cross-validation
rf_scores = cross_val_score(rf_pipeline, X, y, cv=5)
gb_scores = cross_val_score(gb_pipeline, X, y, cv=5)

print(f"RF CV Accuracy: {rf_scores.mean():.4f}")
print(f"GB CV Accuracy: {gb_scores.mean():.4f}")
