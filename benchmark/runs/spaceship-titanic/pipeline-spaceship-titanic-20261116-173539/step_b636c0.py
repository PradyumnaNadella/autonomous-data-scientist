from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']:
    train[col] = train[col].astype(str)
X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported'].astype(bool)
train['Num'] = pd.to_numeric(train['Num'], errors='coerce')

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Define pipeline for tuning
clf = Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])

param_dist = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X, y)

print(f"Best Params: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.4f}")
