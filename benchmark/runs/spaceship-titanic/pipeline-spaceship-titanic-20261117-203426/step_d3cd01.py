import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

# Load
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)
y = train_df['Transported'].astype(int)

# Features
# Keep: Group (str), MemberId (str), Deck (str), Num (str), Side (str), HomePlanet, CryoSleep, Cabin, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Surname
# Drop: PassengerId, Name, Cabin (since split), Transported (target)
features_to_drop = ['PassengerId', 'Name', 'Cabin', 'Transported']
X = train_df.drop(columns=features_to_drop)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# 'Group' and 'Num' are better treated as categorical due to high cardinality/ID nature
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname', 'Group', 'MemberId', 'Num']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Models
models = [
    ('RF', RandomForestClassifier(random_state=42)),
    ('XGB', XGBClassifier(random_state=42, n_estimators=100)),
    ('HistGB', HistGradientBoostingClassifier(random_state=42))
]

for name, model in models:
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
