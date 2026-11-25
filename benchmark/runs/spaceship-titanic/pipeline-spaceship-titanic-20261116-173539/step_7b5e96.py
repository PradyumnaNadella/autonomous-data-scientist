import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Re-define build_features_fixed
def build_features_fixed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = pd.to_numeric(cabin_split[1])
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Name', 'Cabin'])
    return df

df = pd.read_csv('train.csv')
df = build_features_fixed(df)
target = 'Transported'

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'Group']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']

for col in cat_features:
    df[col] = df[col].astype(str)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# XGBoost Tuning
xgb_pipe = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
param_dist = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1, 0.2]
}

search = RandomizedSearchCV(xgb_pipe, param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
search.fit(df.drop(columns=[target, 'PassengerId']), df[target])

print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")
