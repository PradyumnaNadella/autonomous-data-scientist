# Create train_model.py
train_model_script = """
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from features import build_features

def run():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    X = build_features(train).drop(['PassengerId', 'Transported'], axis=1)
    y = train['Transported']
    X_test = build_features(test).drop(['PassengerId'], axis=1)

    numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1))
    ])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X_test)
    output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
    output.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == '__main__':
    run()
"""
with open('train_model.py', 'w') as f:
    f.write(train_model_script)
