
import pandas as pd
import joblib
from features import build_features

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = build_features(train).drop(['Transported', 'PassengerId'], axis=1)
y_train = train['Transported'].astype(int)
X_test = build_features(test).drop('PassengerId', axis=1)

# Preprocess types
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']
for df in [X_train, X_test]:
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[cat_cols] = df[cat_cols].astype(str)

# Load and fit model
model = joblib.load('model.joblib')
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
