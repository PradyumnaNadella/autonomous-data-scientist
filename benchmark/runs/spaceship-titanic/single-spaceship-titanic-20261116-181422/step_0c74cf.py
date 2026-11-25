import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.drop('Transported', axis=1)
y = train['Transported']
X_test = test

def preprocess(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
    return df

X = preprocess(X)
X_test_processed = preprocess(X_test)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
])

xgb_pipe.fit(X, y)
predictions = xgb_pipe.predict(X_test_processed)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
