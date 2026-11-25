import pandas as pd
import joblib

# Use the Logistic Regression model (it had higher CV)
X_train = train.drop(columns=['Transported'])
y_train = train['Transported']
X_test = test

def preprocess(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    return df

# Rebuild the pipeline for Logistic Regression explicitly to be sure
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
final_model.fit(preprocess(X_train), y_train)

# Predict
test_processed = preprocess(X_test)
predictions = final_model.predict(test_processed)

# Create submission
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
