import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def preprocess_df(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(columns=['Cabin', 'Name'], inplace=True)
    for col in ['CryoSleep', 'VIP']:
        df[col] = df[col].astype(str)
    return df

X_train = preprocess_df(train.drop('Transported', axis=1))
y_train = train['Transported']
X_test = preprocess_df(test)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(random_state=42))])

clf.fit(X_train, y_train)
preds = clf.predict(X_test)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
