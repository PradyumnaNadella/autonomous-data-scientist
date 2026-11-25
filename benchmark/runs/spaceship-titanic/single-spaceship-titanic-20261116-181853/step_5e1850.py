import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Re-load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
def preprocess_data(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['Cabin', 'Name'])
    return df

X_train = preprocess_data(train.drop(columns=['Transported']))
y_train = train['Transported']
X_test = preprocess_data(test)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

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

# Best model
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))])

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Output
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
