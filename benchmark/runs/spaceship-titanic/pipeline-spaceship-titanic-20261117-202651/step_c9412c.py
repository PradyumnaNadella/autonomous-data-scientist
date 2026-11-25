import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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

X = train_df.drop(columns=['Transported', 'PassengerId', 'Name'])
y = train_df['Transported'].astype(int)

# 1. Random Forest
rf_pipe = Pipeline([('pre', preprocessor), ('model', RandomForestClassifier(random_state=42))])
rf_score = cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy').mean()
print(f"RF CV Score: {rf_score}")

# 2. LGBM
lgbm_pipe = Pipeline([('pre', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])
lgbm_score = cross_val_score(lgbm_pipe, X, y, cv=5, scoring='accuracy').mean()
print(f"LGBM CV Score: {lgbm_score}")

# 3. Tuned LGBM
tuned_lgbm_pipe = Pipeline([('pre', preprocessor), ('model', LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1))])
tuned_lgbm_score = cross_val_score(tuned_lgbm_pipe, X, y, cv=5, scoring='accuracy').mean()
print(f"Tuned LGBM CV Score: {tuned_lgbm_score}")
