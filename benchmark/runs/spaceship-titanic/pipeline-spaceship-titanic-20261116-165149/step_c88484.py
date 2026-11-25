import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from features import build_features

# Train and Predict
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = build_features(train)
test = build_features(test)

X = train.drop(columns=['Transported'])
y = train['Transported']
X_test = test

cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'CryoSleep', 'VIP']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LGBMClassifier(verbose=-1))])
pipeline.fit(X, y)

preds = pipeline.predict(X_test)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved.")
