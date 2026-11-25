import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features import build_features

# Load data
train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Identify features
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Group, Surname are high cardinality, let's treat carefully or drop if needed
# For now, excluding them to simplify, focus on reliable columns
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Build Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Define models
models = {
    'RF': RandomForestClassifier(random_state=42),
    'XGB': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LGBM': LGBMClassifier(random_state=42)
}

# Evaluate
results = {}
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {results[name]:.4f}")
