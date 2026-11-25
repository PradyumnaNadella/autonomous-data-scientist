# Train with original LightGBM 100
from features import build_features
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
import numpy as np

train = pd.read_csv('train.csv')
train_feat = build_features(train)
y = train_feat['Transported'].astype(int)
X = train_feat.drop('Transported', axis=1)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Fit final model
clf = Pipeline([('preprocessor', preprocessor), ('classifier', LGBMClassifier(n_estimators=100, verbose=-1))])
clf.fit(X, y)

# Get feature importances
importances = clf.named_steps['classifier'].feature_importances_
ohe = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
cat_feature_names = ohe.get_feature_names_out(cat_cols)
all_feature_names = np.concatenate([num_cols, cat_feature_names])

# Sort and print top 5
df_feat_imp = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
print(df_feat_imp.sort_values(by='importance', ascending=False).head(5))

# Generate predictions
test = pd.read_csv('test.csv')
test_feat = build_features(test)
for col in test_feat.select_dtypes(include=['object']).columns:
    test_feat[col] = test_feat[col].astype(str)

preds = clf.predict(test_feat)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
