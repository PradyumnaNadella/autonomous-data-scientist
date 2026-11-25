import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
def feature_eng(df):
    df = df.copy()
    # Extract Group and Cabin features
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(['PassengerId', 'Name', 'Cabin', 'Num'], axis=1)
    return df

X = feature_eng(train.drop('Transported', axis=1))
y = train['Transported']
X_test = feature_eng(test)

# Define Preprocessor
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Models
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

gb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(random_state=42))])

# CV
scores_rf = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
scores_gb = cross_val_score(gb, X, y, cv=5, scoring='accuracy')

print(f"RF CV Score: {scores_rf.mean():.4f}")
print(f"GB CV Score: {scores_gb.mean():.4f}")
