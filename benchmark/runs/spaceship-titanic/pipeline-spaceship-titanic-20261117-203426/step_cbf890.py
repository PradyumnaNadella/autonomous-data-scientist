import pandas as pd
from features import build_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Feature Engineering
train_feat = build_features(train_df)
test_feat = build_features(test_df)

# Drop non-feature columns
drop_cols = ['PassengerId', 'Name']
train_feat = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
# Keep PassengerId for final file
passenger_ids = test_feat['PassengerId']
test_feat = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])

X = train_feat.drop(columns=['Transported'])
y = train_feat['Transported']

def cast_to_string(X):
    return pd.DataFrame(X).astype(str)

categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('cast', FunctionTransformer(cast_to_string)),
                          ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('classifier', HistGradientBoostingClassifier(random_state=42, max_iter=200, learning_rate=0.1))])

# Train
pipeline.fit(X, y)

# Predict
predictions = pipeline.predict(test_feat)

# Save
submission = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
