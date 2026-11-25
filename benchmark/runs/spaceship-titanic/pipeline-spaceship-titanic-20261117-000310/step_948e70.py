from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
import pandas as pd
from features import build_features

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = build_features(train).drop(['PassengerId', 'Transported'], axis=1)
y = train['Transported']
X_test = build_features(test).drop(['PassengerId'], axis=1)

# Identify features
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Final Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1))
])

# Fit on all training data
pipeline.fit(X, y)

# Predict
predictions = pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)

# Feature Importances (requires a bit of work to map back)
# Get the feature names
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
cat_names = list(ohe.get_feature_names_out(categorical_features))
all_features = numeric_features + cat_names

importances = pipeline.named_steps['model'].feature_importances_
feature_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)
print(feature_imp.head(5))
