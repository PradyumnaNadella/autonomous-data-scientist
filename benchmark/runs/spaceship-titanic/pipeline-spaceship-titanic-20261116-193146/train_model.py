
import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Engineering
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported']
X_test = build_features(test_df)
X = X.drop(['PassengerId', 'Name'], axis=1, errors='ignore')
pass_id = test_df['PassengerId']
X_test = X_test.drop(['PassengerId', 'Name'], axis=1, errors='ignore')

# String conversion
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

# Fit and Predict
model = RandomForestClassifier(n_estimators=100, random_state=42)
pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
pipeline.fit(X, y)
preds = pipeline.predict(X_test)

# Save
pd.DataFrame({'PassengerId': pass_id, 'Transported': preds}).to_csv('predictions.csv', index=False)
