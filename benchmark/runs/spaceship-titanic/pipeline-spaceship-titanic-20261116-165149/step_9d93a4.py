import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from features import build_features

train_data = pd.read_csv('train.csv')
X = build_features(train_data.drop(columns=['Transported']))
y = train_data['Transported']

def add_group_size(df):
    df = df.copy()
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    return df

X = add_group_size(X)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Tune GB
gb_pipe_tuned = Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42))])
gb_score_tuned = cross_val_score(gb_pipe_tuned, X, y, cv=5, scoring='accuracy').mean()
print(f"GB Tuned CV Score: {gb_score_tuned}")
