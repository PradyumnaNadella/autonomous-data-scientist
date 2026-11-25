import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = 'Transported'
X = train.drop(columns=[target])
y = train[target].astype(int)

def preprocess(df):
    df = df.copy()
    if 'Cabin' in df.columns:
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df = df.drop(columns=['Cabin'])
    for col in ['CryoSleep', 'VIP']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    df = df.drop(columns=['Name', 'PassengerId'], errors='ignore')
    return df

X_pre = preprocess(X)
X_test = preprocess(test)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

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

xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', XGBClassifier(n_estimators=100, random_state=42))])

xgb_model.fit(X_pre, y)
predictions = xgb_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions.astype(bool)})
output.to_csv('predictions.csv', index=False)
print("Predictions saved.")
