from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
target = 'Transported'

categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])

model.fit(train[features], train[target])

# Predict
test['Transported'] = model.predict(test[features])
submission = test[['PassengerId', 'Transported']]
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
