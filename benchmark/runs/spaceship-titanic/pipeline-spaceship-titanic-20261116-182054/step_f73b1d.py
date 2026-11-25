import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from features import build_features

# Load and prepare
train = build_features(pd.read_csv('train.csv'))
test = build_features(pd.read_csv('test.csv'))

# Convert problematic columns to string to ensure uniformity
for col in ['CryoSleep', 'VIP']:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

X = train.drop('Transported', axis=1)
y = train['Transported']

# Define columns
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Full Pipeline
model = RandomForestClassifier(n_estimators=100, random_state=42)
full_pipeline = Pipeline([('prep', preprocessor), ('clf', model)])

# Train and Predict
full_pipeline.fit(X, y)
predictions = full_pipeline.predict(test)

# Save
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
