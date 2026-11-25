import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering: 
# Split Cabin into deck/num/side
def process_cabin(df):
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    return df.drop(columns=['Cabin'])

train = process_cabin(train)
test = process_cabin(test)

X = train.drop(columns=['Transported', 'PassengerId', 'Name'])
y = train['Transported'].astype(int)
X_test = test.drop(columns=['PassengerId', 'Name'])

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

def eval_model(model, name):
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    return scores.mean()

rf_score = eval_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
print(f"RF CV Score: {rf_score}")

gb_score = eval_model(GradientBoostingClassifier(random_state=42), "Gradient Boosting")
print(f"GB CV Score: {gb_score}")
