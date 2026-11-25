from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

from features import build_features
train = build_features(train)
test = build_features(test)

# Setup Pipeline
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Group', 'Surname', 'CryoSleep', 'VIP']

# Note: Num is categorical in logic, let's treat it as such or drop/impute
categorical_features.append('Num')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Models
models = {
    'RF': RandomForestClassifier(random_state=42),
    'XGB': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# CV
X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print(f"{name} Accuracy: {scores.mean():.4f}")

# Final selection: XGBoost appears stronger based on typical performance on this dataset
final_pipe = Pipeline([('prep', preprocessor), ('model', models['XGB'])])
final_pipe.fit(X, y)

# Predict
preds = final_pipe.predict(test.drop(columns=['PassengerId']))
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
