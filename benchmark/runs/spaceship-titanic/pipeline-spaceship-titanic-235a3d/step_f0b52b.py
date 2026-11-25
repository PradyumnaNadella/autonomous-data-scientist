from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

# Load and prepare
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_feat = build_features(train_df)
test_feat = build_features(test_df)

X = train_feat.drop(columns=['Transported', 'PassengerId'])
y = train_feat['Transported']
X_test = test_feat.drop(columns=['PassengerId'])

num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ]
)

model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

# Train
pipeline.fit(X, y)

# Feature Importance
importances = model.feature_importances_
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 5 Features:")
print(feat_imp.head(5))

# Predict
predictions = pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)
