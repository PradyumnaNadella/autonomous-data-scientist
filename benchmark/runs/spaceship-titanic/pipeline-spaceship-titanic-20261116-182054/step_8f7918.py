from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from features import build_features
import numpy as np

train_df = pd.read_csv('train.csv')
X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported'].astype(int)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = [c for c in X_train.columns if c not in num_cols and c != 'PassengerId']

for col in cat_cols:
    X_train[col] = X_train[col].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

model = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000))])
model.fit(X_train.drop('PassengerId', axis=1), y_train)

# Feature importance (coefficients)
# Need to get feature names after OneHotEncoder
ohe = model.named_steps['pre'].named_transformers_['cat'].named_steps['encoder']
cat_feature_names = ohe.get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

importances = np.abs(model.named_steps['model'].coef_[0])
indices = np.argsort(importances)[::-1]

print("Top 5 Features:")
for i in range(5):
    print(f"{all_feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
