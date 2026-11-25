import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

# Load
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)

for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname', 'Group', 'MemberId', 'Num']:
    train_df[col] = train_df[col].astype(str)

y = train_df['Transported'].astype(int)
features_to_drop = ['PassengerId', 'Name', 'Cabin', 'Transported']
X = train_df.drop(columns=features_to_drop)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname', 'Group', 'MemberId', 'Num']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# Tune HistGB
clf = Pipeline([('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42, learning_rate=0.05, max_iter=200))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"HistGB_tuned: {scores.mean():.4f}")
