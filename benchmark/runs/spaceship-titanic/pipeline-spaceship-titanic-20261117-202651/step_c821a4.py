import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

train_raw = pd.read_csv('train.csv')
train_df = build_features(train_raw)

# Fix types
train_df['CryoSleep'] = train_df['CryoSleep'].astype(str)
train_df['VIP'] = train_df['VIP'].astype(str)

numeric_features = ['Age', 'Num', 'TotalSpend', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# CV test
model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
pipe = Pipeline([('pre', preprocessor), ('model', model)])

X = train_df.drop(columns=['Transported', 'PassengerId', 'Name'])
y = train_df['Transported'].astype(int)

scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f}")
