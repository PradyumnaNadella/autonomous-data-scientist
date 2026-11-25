import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from features import build_features

# Create train_model.py
script = """
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from features import build_features

def add_group_size(df):
    df = df.copy()
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    return df

def train_and_predict():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    y = train_data['Transported']
    X_train = add_group_size(build_features(train_data.drop(columns=['Transported'])))
    X_test = add_group_size(build_features(test_data))
    
    num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']
    cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42))
    ])
    
    pipeline.fit(X_train, y)
    
    preds = pipeline.predict(X_test)
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': preds})
    submission.to_csv('predictions.csv', index=False)
    
    # Feature Importances
    feature_names = num_features + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_features))
    importances = pipeline.named_steps['model'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("Top 5 Feature Importances:")
    print(feat_imp.head(5))

if __name__ == '__main__':
    train_and_predict()
"""

with open('train_model.py', 'w') as f:
    f.write(script)
