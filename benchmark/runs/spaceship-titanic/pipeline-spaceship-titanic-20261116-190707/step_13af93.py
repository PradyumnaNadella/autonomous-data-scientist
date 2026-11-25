import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from features import build_features

train = pd.read_csv('train.csv')
train = build_features(train)

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Num', 'Side', 'Surname']
for col in cat_cols:
    train[col] = train[col].astype(str)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported'].astype(int)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = [col for col in X.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

xgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

param_dist = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 6, 9],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X, y)

print(f"Best Score: {random_search.best_score_}")
print(f"Best Params: {random_search.best_params_}")
