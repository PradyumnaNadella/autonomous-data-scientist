from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from features import build_features

# Load data
train = pd.read_csv('train.csv')
X = build_features(train).drop(['PassengerId', 'Transported'], axis=1)
y = train['Transported']

# Identify features
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tune LGBM
pipeline = Pipeline([('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])
param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [5, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
grid.fit(X, y)
print(f"Best Score: {grid.best_score_:.4f}")
print(f"Best Params: {grid.best_params_}")
