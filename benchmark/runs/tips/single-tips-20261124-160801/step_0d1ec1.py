import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop(['row_id', 'tip'], axis=1)
y_train = train['tip']
X_test = test.drop(['row_id'], axis=1)

cat_cols = ['sex', 'smoker', 'day', 'time']
num_cols = ['total_bill', 'size']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

submission = pd.DataFrame({'row_id': test['row_id'], 'tip': preds})
submission.to_csv('submission.csv', index=False)
print(submission.head())
