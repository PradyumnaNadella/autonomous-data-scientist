import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Rebuild model_gb
train = pd.read_csv('train.csv')
train_df = build_features(train)
test = pd.read_csv('test.csv')
test_df = build_features(test)

cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_features),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

model_gb = Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
model_gb.fit(train_df.drop(columns=['PassengerId', 'Transported']), train_df['Transported'])

predictions = model_gb.predict(test_df.drop(columns=['PassengerId']))
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)
print("Done.")
