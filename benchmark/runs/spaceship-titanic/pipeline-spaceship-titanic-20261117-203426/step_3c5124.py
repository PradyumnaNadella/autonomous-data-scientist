import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from features import build_features

# Load and prepare
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)

y = train_df['Transported'].astype(int)
features_to_drop = ['PassengerId', 'Name', 'Cabin', 'Transported']
X = train_df.drop(columns=features_to_drop)

# Cast types for processing
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname', 'Group', 'MemberId', 'Num']

for col in categorical_features:
    X[col] = X[col].astype(str)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# CV
model = Pipeline([('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Mean CV Accuracy: {scores.mean():.4f}")
