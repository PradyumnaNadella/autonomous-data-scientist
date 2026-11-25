import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load
train = pd.read_csv('train.csv')
train = build_features(train)
X = train.drop(columns=['Transported', 'PassengerId', 'Group', 'Name'])
y = train['Transported'].astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['Deck', 'Side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
    ]
)

model = Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Score: {scores.mean():.4f}")
