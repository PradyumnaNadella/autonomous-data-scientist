# Since HistGradientBoosting failed with a timeout but RandomForest is working,
# and OHE on 'GroupId' might be too large, let's refine features by dropping GroupId.
# Then fit the best model on the full train and predict.
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop GroupId to speed up OHE
def build_features_fast(df):
    df = build_features(df)
    return df.drop(columns=['GroupId'])

train_f = build_features_fast(train)
test_f = build_features_fast(test)

X = train_f.drop(columns=['Transported', 'PassengerId'])
y = train_f['Transported']
X_test = test_f.drop(columns=['PassengerId'])

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Num', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))])
clf.fit(X, y)
predictions = clf.predict(X_test)

submission = pd.DataFrame({'PassengerId': test_f['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
