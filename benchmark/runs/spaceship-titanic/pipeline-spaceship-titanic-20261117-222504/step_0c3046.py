with open('train_model.py', 'w') as f:
    f.write('''
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
passenger_ids = test['PassengerId']

train_f = build_features(train)
test_f = build_features(test)

y = train_f['Transported']
X_train = train_f.drop(columns=['Transported', 'PassengerId'])
X_test = test_f.drop(columns=['PassengerId'])

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
num_features = X_train.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=5, n_estimators=200, random_state=42, verbose=-1)
pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
pipeline.fit(X_train, y)

preds = pipeline.predict(X_test)
submission = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
''')
