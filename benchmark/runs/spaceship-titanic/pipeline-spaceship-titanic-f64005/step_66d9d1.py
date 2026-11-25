import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from features import build_features

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
passenger_ids = test['PassengerId']

# Process
train = build_features(train)
test = build_features(test)

for col in train.select_dtypes(include=['object', 'bool']).columns:
    if col != 'Transported':
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

y = train['Transported']
X = train.drop('Transported', axis=1)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Chosen Model
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
pipeline.fit(X, y)

# Predictions
predictions = pipeline.predict(test)
output = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predictions})
output.to_csv('predictions.csv', index=False)

# Feature Importance
importances = model.feature_importances_
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp.head(5))
