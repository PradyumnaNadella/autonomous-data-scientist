import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Feature Engineering
train_df = build_features(train_df)
test_df = build_features(test_df)

# Drop irrelevant
drop_cols = ['PassengerId', 'Name', 'Cabin']
X = train_df.drop(columns=drop_cols + ['Transported'])
y = train_df['Transported'].astype(int)

# Identify feature types
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# Compare Models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'HistGBM': HistGradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f'{name} CV Accuracy: {scores.mean():.4f}')
