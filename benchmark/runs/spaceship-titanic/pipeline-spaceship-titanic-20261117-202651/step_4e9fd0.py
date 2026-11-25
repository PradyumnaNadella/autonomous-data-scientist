import pandas as pd
import numpy as np
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

train_df = build_features(pd.read_csv('train.csv'))
for col in ['CryoSleep', 'VIP']:
    train_df[col] = train_df[col].astype(str)

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

model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
pipe = Pipeline([('pre', preprocessor), ('model', model)])

X = train_df.drop(columns=['Transported', 'PassengerId', 'Name'])
y = train_df['Transported'].astype(int)
pipe.fit(X, y)

# Get importances
ohe = pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
cat_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, cat_names])
importances = pipe.named_steps['model'].feature_importances_

df_importances = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
print(df_importances.sort_values(by='importance', ascending=False).head(5))
