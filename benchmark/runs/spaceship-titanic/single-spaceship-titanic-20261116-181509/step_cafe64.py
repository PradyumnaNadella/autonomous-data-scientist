import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)

# Feature engineering: split Cabin
for df in [X]:
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

# Select features
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

# Model 1: Logistic Regression
clf_lr = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter=1000))])

# Model 2: Random Forest
clf_rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

cv_scores_lr = cross_val_score(clf_lr, X, y, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(clf_rf, X, y, cv=5, scoring='accuracy')

print(f"LR CV Score: {cv_scores_lr.mean():.4f}")
print(f"RF CV Score: {cv_scores_rf.mean():.4f}")
