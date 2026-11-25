import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported']

# Feature Engineering: Extract deck/num/side from Cabin
def preprocess_cabin(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    return df.drop('Cabin', axis=1)

X = preprocess_cabin(X)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

# Model 1: RandomForest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

# Model 2: GradientBoosting
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GradientBoostingClassifier(random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_score = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='accuracy').mean()
gb_score = cross_val_score(gb_pipeline, X, y, cv=cv, scoring='accuracy').mean()

print(f"Random Forest CV: {rf_score:.4f}")
print(f"Gradient Boosting CV: {gb_score:.4f}")
