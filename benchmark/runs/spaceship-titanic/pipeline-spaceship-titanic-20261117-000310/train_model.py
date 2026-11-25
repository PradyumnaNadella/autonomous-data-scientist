
import pandas as pd
import joblib
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

y = train_df['Transported'].astype(int)
X = build_features(train_df.drop('Transported', axis=1)).drop('PassengerId', axis=1)
X_test = build_features(test_df.copy()).drop('PassengerId', axis=1)

# Define pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('classifier', LGBMClassifier(learning_rate=0.05, max_depth=10, n_estimators=100, random_state=42, verbose=-1))])

# Train on full set
model.fit(X, y)

# Predict
predictions = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': predictions.astype(bool)
})
submission.to_csv('predictions.csv', index=False)
