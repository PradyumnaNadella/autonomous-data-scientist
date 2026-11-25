import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from features import build_features

# Load and prepare
train = build_features(pd.read_csv('train.csv'))
X = train.drop('Transported', axis=1)
y = train['Transported']

# Define columns
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

def log_experiment(name, model, X, y):
    pipeline = Pipeline([('prep', preprocessor), ('clf', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Experiments
log_experiment('LogisticRegression', LogisticRegression(max_iter=1000), X, y)
log_experiment('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42), X, y)
