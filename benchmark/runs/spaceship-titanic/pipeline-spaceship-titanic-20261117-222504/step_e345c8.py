import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import build_features

train = build_features(pd.read_csv('train.csv'))
test = build_features(pd.read_csv('test.csv'))

X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']
X_test = test.drop(columns=['PassengerId'])

# Ensure consistent types
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

final_model = Pipeline([('pre', preprocessor), ('clf', XGBClassifier(eval_metric='logloss', random_state=42))])
final_model.fit(X, y)
predictions = final_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
