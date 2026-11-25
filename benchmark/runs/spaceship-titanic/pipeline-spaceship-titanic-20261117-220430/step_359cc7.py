# Tune GBM
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.read_csv('train.csv')
train = build_features(train)
X = train.drop(columns=['Transported', 'PassengerId', 'Group'])
y = train['Transported'].astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['Deck', 'Side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
    ]
)

pipeline = Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingClassifier(random_state=42))])
params = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
search = RandomizedSearchCV(pipeline, params, cv=5, n_iter=3, scoring='accuracy', random_state=42)
search.fit(X, y)
print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")
