# Check if the modelers' predictions were generated with the reported model.
# I will re-run the pipeline to check if I get the same format.
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb
from features import build_features
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = build_features(train.drop('Transported', axis=1))
y_train = train['Transported']
X_test = build_features(test)

# I need to know how they handled categorical features (NaNs)
# My guess is OneHotEncoder with handle_unknown='ignore'
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group', 'Surname']

# Reconstruct the pipeline they *likely* used based on the problem
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Train on full
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100))])

clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(preds[:5])
print(type(preds[0]))
