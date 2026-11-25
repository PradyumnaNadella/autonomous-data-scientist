import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
for df in [train, test]:
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    df['Deck'] = df['Cabin'].str.split('/', expand=True)[0]
    df['Side'] = df['Cabin'].str.split('/', expand=True)[2]
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].fillna(0).sum(axis=1)
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']:
        df[col] = df[col].astype(str)

features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Deck', 'Side', 'TotalSpend', 'Group']
numeric_features = ['Age', 'TotalSpend']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']

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

final_model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

X = train.drop('Transported', axis=1)
y = train['Transported']
final_model.fit(X, y)

predictions = final_model.predict(test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
