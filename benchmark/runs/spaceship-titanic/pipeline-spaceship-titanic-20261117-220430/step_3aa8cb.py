import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from features import build_features

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = build_features(train)
test_processed = build_features(test)

X = train.drop(columns=['Transported', 'PassengerId', 'Name', 'Group'])
y = train['Transported']

# Final Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

model = Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
model.fit(X, y)

# Predict
test_X = test_processed.drop(columns=['PassengerId', 'Name', 'Group'])
preds = model.predict(test_X)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
