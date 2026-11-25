import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib

# Load
df = pd.read_csv('train.csv')
features = build_features(df)
X = features.drop(columns=['Transported', 'PassengerId'])
y = features['Transported'].astype(int)

# Force all categorical columns to string
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype(str)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Chosen Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

model.fit(X, y)
joblib.dump(model, 'model.joblib')

# Feature Importance
feature_names = numeric_features + list(model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
importances = model.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 5 Features:")
print(feat_imp.head(5))

# Predict
test_df = pd.read_csv('test.csv')
test_features = build_features(test_df)
# Ensure consistent string types
for col in test_features.columns:
    if test_features[col].dtype == 'object' or test_features[col].dtype == 'bool':
        test_features[col] = test_features[col].astype(str)

test_X = test_features.drop(columns=['PassengerId'])
preds = model.predict(test_X)
submission = pd.DataFrame({'PassengerId': test_features['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
