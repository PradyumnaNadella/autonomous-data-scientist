# Prepare train_model.py
script = """
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from lightgbm import LGBMClassifier
from features import build_features

def cast_to_str(df):
    return df.astype(str)

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    X = build_features(train_df.drop('Transported', axis=1))
    y = train_df['Transported'].astype(int)
    X_test = build_features(test_df)
    
    numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    
    cat_pipeline = Pipeline([
        ('cast', FunctionTransformer(cast_to_str)),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1))
    ])
    
    model.fit(X, y)
    preds = model.predict(X_test)
    
    output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds.astype(bool)})
    output.to_csv('predictions.csv', index=False)
    print("Predictions saved.")

if __name__ == '__main__':
    main()
"""
with open('train_model.py', 'w') as f:
    f.write(script)
print("train_model.py created.")
