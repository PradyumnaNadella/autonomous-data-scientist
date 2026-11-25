import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

df = pd.read_csv('train.csv')
features = build_features(df)
X = features.drop(columns=['Transported', 'PassengerId'])
y = features['Transported'].astype(int)

# Force all categorical columns to string
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype(str)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

xgb_pipe = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(xgb_pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)

print(f"Best CV Score: {grid.best_score_:.4f}")
print(f"Best Params: {grid.best_params_}")
