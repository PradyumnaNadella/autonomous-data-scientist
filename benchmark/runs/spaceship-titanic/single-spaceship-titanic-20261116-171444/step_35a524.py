import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Re-load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def preprocess_data(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
    return df

train_df = preprocess_data(train)
test_df = preprocess_data(test)

X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int)

# Identify features
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Build pipeline
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

# Final Model
model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', GradientBoostingClassifier(random_state=42))])

model.fit(X, y)

# Predict
test_X = test_df
predictions = model.predict(test_X)

# Format for submission
submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                           'Transported': predictions.astype(bool)})

submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
