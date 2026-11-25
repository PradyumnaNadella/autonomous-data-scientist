import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    df = df.copy()
    # Feature engineering: extract Group and Deck/Side from Cabin
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(float)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(columns=['Name', 'Num', 'Cabin'], inplace=True)
    
    # Fill missing numeric values
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Group']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill missing categorical values
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        # Ensure they are strings for OneHotEncoder
        df[col] = df[col].astype(str)
        
    return df

train = pd.read_csv('train.csv')
train = preprocess_data(train)

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

# Identify numeric vs categorical columns
# Numeric were converted to float/int, categorical are object
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f}")
