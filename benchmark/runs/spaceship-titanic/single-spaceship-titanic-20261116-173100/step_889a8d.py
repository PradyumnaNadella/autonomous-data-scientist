import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.drop('Transported', axis=1)
y = train['Transported']
X_test = test

# Preprocessing
def preprocess_features(df):
    df = df.copy()
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    return df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

X_eng = preprocess_features(X)
X_test_eng = preprocess_features(X_test)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Best Model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter=1000))])
clf.fit(X_eng, y)

# Predictions
predictions = clf.predict(X_test_eng)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
