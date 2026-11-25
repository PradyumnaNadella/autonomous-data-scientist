import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)
X = train_df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Transported'])
y = train_df['Transported'].astype(int)

# Preprocessing
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']:
    X[col] = X[col].astype(str)
for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']:
    X[col] = pd.to_numeric(X[col], errors='coerce')

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Side', 'Surname']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))]), cat_features)
])

# Tuning
pipe = Pipeline([('preprocessor', preprocessor), ('model', HistGradientBoostingClassifier(random_state=42))])
params = {'model__learning_rate': [0.05, 0.1], 'model__max_depth': [5, 10]}
grid = GridSearchCV(pipe, params, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)
print(f'Best CV: {grid.best_score_:.4f}')
print(f'Best Params: {grid.best_params_}')
