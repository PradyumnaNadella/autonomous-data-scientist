import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Prepare data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop('Transported', axis=1)
y_train = train['Transported']
X_test = test

# Feature engineering
for df in [X_train, X_test]:
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

final_clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])

final_clf.fit(X_train, y_train)

# Predict
predictions = final_clf.predict(X_test)

# Create submission
submission = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
