import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def preprocess(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    return df

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

final_model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
final_model.fit(preprocess(train.drop(columns=['Transported'])), train['Transported'])

# Predict
test_processed = preprocess(test)
predictions = final_model.predict(test_processed)

# Create submission
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
