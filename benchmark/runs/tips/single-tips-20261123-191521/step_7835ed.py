import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(['row_id', 'tip'], axis=1)
y = train['tip']

numeric_features = ['total_bill', 'size']
categorical_features = ['sex', 'smoker', 'day', 'time']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

final_model.fit(X, y)

X_test = test.drop(['row_id'], axis=1)
preds = final_model.predict(X_test)

submission = pd.DataFrame({'row_id': test['row_id'], 'tip': preds})
submission.to_csv('predictions.csv', index=False)
print(submission.head())
