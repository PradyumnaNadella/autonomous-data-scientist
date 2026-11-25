import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Load
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)

# Fix types
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname', 'Group', 'MemberId', 'Num']:
    train_df[col] = train_df[col].astype(str)

y = train_df['Transported'].astype(int)
features_to_drop = ['PassengerId', 'Name', 'Cabin', 'Transported']
X = train_df.drop(columns=features_to_drop)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Remove 'Surname' and 'Group' as they have high cardinality and may be slowing things down
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'MemberId', 'Num']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

models = [
    ('RF', RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=50)),
    ('HistGB', HistGradientBoostingClassifier(random_state=42))
]

for name, model in models:
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
