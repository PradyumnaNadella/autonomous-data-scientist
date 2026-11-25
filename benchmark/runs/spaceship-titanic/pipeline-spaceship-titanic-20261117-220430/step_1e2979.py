# Create train_model.py
with open('train_model.py', 'w') as f:
    f.write(r"""
import pandas as pd
from features import build_features
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = build_features(train)
test = build_features(test)

X = train.drop(columns=['Transported', 'PassengerId', 'Group'])
y = train['Transported'].astype(int)
X_test = test.drop(columns=['PassengerId', 'Group'])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['Deck', 'Side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
    ]
)

model = Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))])
model.fit(X, y)
preds = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds.astype(bool)})
output.to_csv('predictions.csv', index=False)
print("Predictions saved.")
""")
