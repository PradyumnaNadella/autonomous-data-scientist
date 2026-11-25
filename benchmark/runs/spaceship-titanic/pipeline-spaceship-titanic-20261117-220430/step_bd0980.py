import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Re-setup
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)
X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported'].astype(int)

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
    ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

param_dist = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1]
}

search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
search.fit(X, y)
print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")
