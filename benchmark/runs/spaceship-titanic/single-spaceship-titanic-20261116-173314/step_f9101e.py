import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process_data(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(['PassengerId', 'Cabin', 'Name', 'Num'], axis=1)
    return df

X = process_data(train.drop('Transported', axis=1))
y = train['Transported'].astype(int)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)

X_test = process_data(test)
predictions = pipeline.predict(X_test)

final_df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions.astype(bool)})
final_df.to_csv('predictions.csv', index=False)
print(final_df.head())
