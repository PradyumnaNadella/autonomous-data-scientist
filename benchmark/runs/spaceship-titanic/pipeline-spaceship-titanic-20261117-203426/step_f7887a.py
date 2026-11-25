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
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']
for col in cat_cols:
    train[col] = train[col].fillna('missing').astype(str)

X = train.drop(columns=['Transported'])
y = train['Transported'].astype(int)

# Preprocessing
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
])

# Models
# Use a lighter RandomForest for speed
model_rf = Pipeline([('prep', preprocessor), ('model', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))])
model_hgb = Pipeline([('prep', preprocessor), ('model', HistGradientBoostingClassifier(random_state=42))])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in [("RandomForest", model_rf), ("HistGradientBoosting", model_hgb)]:
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")

# Train best model
model_hgb.fit(X, y)

test = pd.read_csv('test.csv')
test_f = build_features(test)
for col in cat_cols:
    test_f[col] = test_f[col].fillna('missing').astype(str)

preds = model_hgb.predict(test_f)
submission = pd.DataFrame({'PassengerId': test_f['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
