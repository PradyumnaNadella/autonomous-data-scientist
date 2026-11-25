import pandas as pd
import numpy as np
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

for col in X.select_dtypes(include=['object', 'bool']).columns:
    X[col] = X[col].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Fit again for feature importance
clf = XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
pipe.fit(X, y)

# Get feature names after OHE
ohe = pipe.named_steps['prep'].named_transformers_['cat'].named_steps['ohe']
cat_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_names)

importances = pipe.named_steps['clf'].feature_importances_
feature_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

print("Top 5 Features:")
print(feature_imp.head(5))
