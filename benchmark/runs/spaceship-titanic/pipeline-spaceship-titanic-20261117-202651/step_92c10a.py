import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

# Prepare training data
train = build_features(pd.read_csv('train.csv'))
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
for col in cat_cols:
    train[col] = train[col].astype(str)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']
num_cols = X.select_dtypes(include=['float64']).columns.tolist()

# Define and fit model
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])
model = Pipeline([('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])
model.fit(X, y)

# Predict test
test = build_features(pd.read_csv('test.csv'))
for col in cat_cols:
    test[col] = test[col].astype(str)

predictions = model.predict(test.drop(columns=['PassengerId']))
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
