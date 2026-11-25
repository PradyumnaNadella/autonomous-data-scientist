import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Reload data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocessing
def feature_eng(df):
    df = df.copy()
    for col in ['CryoSleep', 'VIP']:
        df[col] = df[col].astype(str)
    
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(['PassengerId', 'Name', 'Cabin', 'Num'], axis=1)
    return df

X = feature_eng(train.drop('Transported', axis=1))
y = train['Transported']
X_test = feature_eng(test)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(random_state=42))])

model.fit(X, y)
preds = model.predict(X_test)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved.")
