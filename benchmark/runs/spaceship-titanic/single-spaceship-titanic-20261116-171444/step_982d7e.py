import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
def preprocess_data(df):
    df = df.copy()
    # Cabin: split into Deck, Num, Side
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
    return df

train_df = preprocess_data(train)
test_df = preprocess_data(test)

X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int)

# Identify features
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
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

gb_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', GradientBoostingClassifier(random_state=42))])

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_score = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy').mean()
gb_score = cross_val_score(gb_model, X, y, cv=cv, scoring='accuracy').mean()

print(f"Random Forest CV Score: {rf_score}")
print(f"Gradient Boosting CV Score: {gb_score}")
