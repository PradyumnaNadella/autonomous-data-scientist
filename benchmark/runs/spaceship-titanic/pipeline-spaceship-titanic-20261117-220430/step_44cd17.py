# Get feature importances
import pandas as pd
from features import build_features
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.read_csv('train.csv')
train = build_features(train)
X = train.drop(columns=['Transported', 'PassengerId', 'Group'])
y = train['Transported'].astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['Deck', 'Side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
    ]
)
model = Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))])
model.fit(X, y)

# Extract feature names
ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
cat_names = ohe.get_feature_names_out(['Deck', 'Side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
all_names = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'] + list(cat_names)

importances = model.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=all_names).sort_values(ascending=False)
print(feat_imp.head(5))
