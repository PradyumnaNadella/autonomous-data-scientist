
import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

# Load
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df = build_features(train_df)
test_df = build_features(test_df)

# Prepare
y = train_df['Transported'].astype(int)
features_to_drop = ['PassengerId', 'Name', 'Cabin', 'Transported']
X = train_df.drop(columns=features_to_drop)
X_test = test_df.drop(columns=['PassengerId', 'Name', 'Cabin'])

for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname', 'Group', 'MemberId', 'Num']:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Surname', 'Group', 'MemberId', 'Num']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# Pipeline
model = Pipeline([('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))])
model.fit(X, y)

# Predict
preds = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds.astype(bool)})
output.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
