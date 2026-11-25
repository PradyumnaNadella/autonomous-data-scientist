import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from lightgbm import LGBMClassifier
from features import build_features
import numpy as np

# Need to refit to get importances
train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

def cast_to_str(df): return df.astype(str)
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
cat_pipeline = Pipeline([
    ('cast', FunctionTransformer(cast_to_str)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric_features),
    ('cat', cat_pipeline, categorical_features)
])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1))
])
model.fit(X, y)

# Extract importances
ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_names = ohe.get_feature_names_out(categorical_features)
all_names = np.concatenate([numeric_features, cat_names])
importances = model.named_steps['classifier'].feature_importances_
top_5_idx = np.argsort(importances)[-5:][::-1]
for i in top_5_idx:
    print(f"{all_names[i]}: {importances[i]}")
