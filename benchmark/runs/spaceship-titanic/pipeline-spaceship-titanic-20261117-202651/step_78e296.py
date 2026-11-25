import pandas as pd
import numpy as np
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

# Prepare data
train_raw = pd.read_csv('train.csv')
train_df = build_features(train_raw)
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

model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
pipe = Pipeline([('pre', preprocessor), ('model', model)])
X = train_df.drop(columns=['Transported', 'PassengerId', 'Name'])
y = train_df['Transported'].astype(int)

pipe.fit(X, y)

# Feature Importance
ohe = pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

importances = pipe.named_steps['model'].feature_importances_
feature_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

print("Top 5 Feature Importances:")
print(feature_imp.head(5))
