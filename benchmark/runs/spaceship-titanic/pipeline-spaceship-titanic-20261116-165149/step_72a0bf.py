import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from features import build_features

train_data = pd.read_csv('train.csv')
X = build_features(train_data.drop(columns=['Transported']))
y = train_data['Transported']

# Identify features
# Drop high cardinality and IDs
# Use: HomePlanet, CryoSleep, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Deck, Side
# GroupSize is important - calculate it
def add_group_size(df):
    df = df.copy()
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    return df

X = add_group_size(X)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Model 1: RandomForest
rf_pipe = Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
rf_score = cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy').mean()
print(f"RF CV Score: {rf_score}")

# Model 2: GradientBoosting
gb_pipe = Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingClassifier(random_state=42))])
gb_score = cross_val_score(gb_pipe, X, y, cv=5, scoring='accuracy').mean()
print(f"GB CV Score: {gb_score}")
