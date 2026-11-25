import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from features import build_features

# Load and prepare
train_df = build_features(pd.read_csv('train.csv'))
test_df = build_features(pd.read_csv('test.csv'))

X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported'].astype(int)
X_test = test_df.drop(columns=['PassengerId'])

# Convert all non-numeric to string
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == 'bool' or df[col].dtype == 'object':
            df[col] = df[col].astype(str)

# Simple pipeline: Just use HistGradientBoosting with OneHotEncoder is too slow,
# let's try just numeric features for a quick submission
# No, let's use the simplest pipeline possible that works:
numeric_features = X.select_dtypes(include=['number']).columns.tolist()

# Just numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))])
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
