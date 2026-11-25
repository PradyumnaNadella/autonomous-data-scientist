
import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

def train_and_predict():
    train_raw = pd.read_csv('train.csv')
    test_raw = pd.read_csv('test.csv')
    
    train_df = build_features(train_raw)
    test_df = build_features(test_raw)
    
    # Fix types
    for df in [train_df, test_df]:
        df['CryoSleep'] = df['CryoSleep'].astype(str)
        df['VIP'] = df['VIP'].astype(str)
        
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
    
    # Predict
    X_test = test_df.drop(columns=['PassengerId', 'Name'])
    preds = pipe.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': preds.astype(bool)
    })
    submission.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == '__main__':
    train_and_predict()
