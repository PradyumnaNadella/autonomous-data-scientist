import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)
test_df = pd.read_csv('test.csv')
test_df = build_features(test_df)

X = train_df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Transported'])
y = train_df['Transported'].astype(int)
X_test = test_df.drop(columns=['PassengerId', 'Name', 'Cabin'])

# Cast types
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))]), cat_features)
])

# Final Pipeline
model = HistGradientBoostingClassifier(random_state=42)
pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])

# Train and Predict
pipe.fit(X, y)
preds = pipe.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
