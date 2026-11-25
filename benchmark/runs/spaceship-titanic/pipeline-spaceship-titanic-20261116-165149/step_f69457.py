import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

# Load and process
train = pd.read_csv('train.csv')
df = build_features(train.drop('Transported', axis=1))
df['Transported'] = train['Transported']
df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Tune LightGBM slightly
model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = df.drop(['Transported', 'PassengerId', 'Group', 'Member', 'Num', 'Surname'], axis=1)
y = df['Transported']

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"Tuned LightGBM: {scores.mean():.4f}")
