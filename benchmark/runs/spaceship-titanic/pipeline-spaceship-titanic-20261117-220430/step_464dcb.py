from features import build_features
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
X_train = train_df.drop('Transported', axis=1)
y_train = train_df['Transported'].astype(int)
X_test = test_df

X_train = build_features(X_train)
X_test = build_features(X_test)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Final model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42))
])

model.fit(X_train, y_train)
preds = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved.")
