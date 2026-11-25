import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from features import build_features

# Load and prepare
train = pd.read_csv('train.csv')
train = build_features(train)
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']:
    train[col] = train[col].astype(str)

X = train.drop(columns=['Transported'])
y = train['Transported'].astype(int)

# Preprocessing
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
])

# Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")

# Train and predict
best_model = Pipeline([('prep', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
best_model.fit(X, y)

test = pd.read_csv('test.csv')
test_f = build_features(test)
for col in cat_cols:
    test_f[col] = test_f[col].astype(str)

preds = best_model.predict(test_f)
submission = pd.DataFrame({'PassengerId': test_f['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
