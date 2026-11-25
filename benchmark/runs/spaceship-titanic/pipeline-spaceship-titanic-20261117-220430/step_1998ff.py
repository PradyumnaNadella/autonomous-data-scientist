# Creating the final train_model.py script as required
script_content = """
import pandas as pd
import joblib
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def train_and_predict():
    # Load Data
    df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Preprocessing
    features = build_features(df)
    X = features.drop(columns=['Transported', 'PassengerId'])
    y = features['Transported'].astype(int)
    
    test_features = build_features(test_df)
    test_X = test_features.drop(columns=['PassengerId'])
    
    # Consistent type forcing
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'bool':
            X[col] = X[col].astype(str)
            test_X[col] = test_X[col].astype(str)
            
    numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    
    model.fit(X, y)
    
    # Predict
    preds = model.predict(test_X)
    submission = pd.DataFrame({'PassengerId': test_features['PassengerId'], 'Transported': preds.astype(bool)})
    submission.to_csv('predictions.csv', index=False)
    
    joblib.dump(model, 'model.joblib')

if __name__ == '__main__':
    train_and_predict()
"""

with open('train_model.py', 'w') as f:
    f.write(script_content)

import os
os.system('python train_model.py')
