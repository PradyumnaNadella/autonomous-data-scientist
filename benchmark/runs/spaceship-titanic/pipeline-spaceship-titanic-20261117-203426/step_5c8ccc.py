import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Load and process
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = build_features(train)
test = build_features(test)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

# Define preprocessor
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

def log_experiment(name, score):
    print(f"EXPERIMENT: {name}, CV Accuracy: {score:.4f}")

# Compare Models
for model, name in [(RandomForestClassifier(random_state=42), 'RandomForest'), 
                    (HistGradientBoostingClassifier(random_state=42), 'HistGB')]:
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=5)
    log_experiment(name, scores.mean())

# Finalize Best
best_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))])
best_clf.fit(X, y)
predictions = best_clf.predict(test.drop(columns=['PassengerId']))
pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions}).to_csv('predictions.csv', index=False)
