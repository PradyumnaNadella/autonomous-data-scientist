# Get top 5 features
import pandas as pd
import joblib
from features import build_features

train = pd.read_csv('train.csv')
X = build_features(train.drop(columns=['Transported']))
model = joblib.load('model.joblib')

# Get feature names after encoding
cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_enc = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
feature_names = list(num_features) + list(cat_enc.get_feature_names_out(cat_features))

# Importance
importances = model.named_steps['classifier'].feature_importances_
feature_imp = pd.Series(importances, index=feature_names)
print(feature_imp.nlargest(5))
