import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(float)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(columns=['Name', 'Num', 'Cabin'], inplace=True)
    
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Group']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(str)
        
    return df

# Load and prepare
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = preprocess_data(train)
test = preprocess_data(test)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']
X_test_final = test.drop(columns=['PassengerId'])

# Setup
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Fit and Predict
model.fit(X, y)
predictions = model.predict(X_test_final)

# Save
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': predictions
})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.")
