import pandas as pd
import numpy as np
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

train_df = pd.read_csv('train.csv')
train_df_f = build_features(train_df)
for col in ['CryoSleep', 'VIP']:
    train_df_f[col] = train_df_f[col].astype(str)
X = train_df_f.drop(columns=['Transported', 'PassengerId'], errors='ignore')
y = train_df_f['Transported'].astype(int)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

final_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1))])
final_model.fit(X, y)

classifier = final_model.named_steps['classifier']
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_names)

importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 features:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]}")
