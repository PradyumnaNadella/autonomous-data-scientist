import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load data
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)
X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported'].astype(int)

# Identify feature types
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Pipeline
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

# Compare models
models = [
    ('RF', RandomForestClassifier(random_state=42)),
    ('XGB', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
