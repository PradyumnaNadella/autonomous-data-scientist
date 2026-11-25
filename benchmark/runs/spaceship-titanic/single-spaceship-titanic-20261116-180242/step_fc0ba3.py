import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Re-prepare data for final training
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def feature_engineering(df):
    df = df.copy()
    # Ensure all categorical columns are string for consistent processing
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        df[col] = df[col].astype(str)
        
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(columns=['Cabin', 'Name', 'PassengerId'], inplace=True)
    return df

X = feature_engineering(train.drop(columns=['Transported']))
y = train['Transported']
X_test = feature_engineering(test)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Ensure numeric columns are actually numeric
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Best Model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', GradientBoostingClassifier(random_state=42))])

model.fit(X, y)
predictions = model.predict(X_test)

# Save predictions
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved.")
