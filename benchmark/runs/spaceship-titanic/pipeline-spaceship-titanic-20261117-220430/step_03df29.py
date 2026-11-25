# Write train_model.py
train_model_content = """
import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib

# Load
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df = build_features(train_df)
test_processed = build_features(test_df)

X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported'].astype(int)
X_test = test_processed.drop(columns=['PassengerId'])

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42))
])

model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_processed['PassengerId'], 'Transported': predictions.astype(bool)})
output.to_csv('predictions.csv', index=False)
"""
with open('train_model.py', 'w') as f:
    f.write(train_model_content)
