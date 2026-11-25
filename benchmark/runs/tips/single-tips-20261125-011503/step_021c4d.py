import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.drop(['row_id', 'tip'], axis=1)
y = train['tip']
X_test = test.drop(['row_id'], axis=1)

cat_features = ['sex', 'smoker', 'day', 'time']
num_features = ['total_bill', 'size']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# Ridge gave best CV score
model = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge())])
model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame({'row_id': test['row_id'], 'tip': predictions})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved.")
