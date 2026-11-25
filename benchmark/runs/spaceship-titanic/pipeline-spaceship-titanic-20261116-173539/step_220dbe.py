# Final pipeline creation and submission
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import build_features

train = build_features(pd.read_csv('train.csv'))
test = build_features(pd.read_csv('test.csv'))

# Preprocessing consistency
for col in ['CryoSleep', 'VIP']:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

train['Num'] = pd.to_numeric(train['Num'], errors='coerce')
test['Num'] = pd.to_numeric(test['Num'], errors='coerce')

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']
X_test = test.drop(columns=['PassengerId'])

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Use XGB as it performed slightly better
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBClassifier(n_estimators=100, random_state=42))])
pipeline.fit(X, y)

predictions = pipeline.predict(X_test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
