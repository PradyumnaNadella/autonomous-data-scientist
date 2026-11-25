import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features import build_features

# Load and prepare data
train = pd.read_csv('train.csv')
train = build_features(train)

X = train.drop(columns=['Transported'])
y = train['Transported']

# Define preprocessor
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'CryoSleep', 'VIP']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Compare models
models = {
    'XGB': XGBClassifier(eval_metric='logloss'),
    'LGBM': LGBMClassifier(verbose=-1)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()

print(f"Results: {results}")
