with open('train_model.py', 'w') as f:
    f.write("""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from features import build_features

# Load and prepare
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = 'Transported'
features_to_drop = ['PassengerId', target]

# Preprocessing
def prepare_data(df):
    df = build_features(df)
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']:
        df[col] = df[col].astype(str)
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')
    return df

train_prepped = prepare_data(train)
test_prepped = prepare_data(test)

X = train_prepped.drop(columns=['PassengerId', target])
y = train_prepped[target].astype(bool)
X_test = test_prepped.drop(columns=['PassengerId'])

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Group']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Final model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss'))
])

model.fit(X, y)
predictions = model.predict(X_test)

# Save
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
""")
