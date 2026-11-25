import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)

# Feature engineering: split cabin
def process_data(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(['PassengerId', 'Cabin', 'Name', 'Num'], axis=1)
    return df

X_processed = process_data(X)

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

# Model 1: Logistic Regression
logreg_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Model 2: RandomForest
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# CV
scores_lr = cross_val_score(logreg_pipe, X_processed, y, cv=5, scoring='accuracy')
scores_rf = cross_val_score(rf_pipe, X_processed, y, cv=5, scoring='accuracy')

print(f"LR CV Score: {scores_lr.mean():.4f}")
print(f"RF CV Score: {scores_rf.mean():.4f}")
