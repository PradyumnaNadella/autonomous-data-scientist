import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported']
X_test = build_features(test_df)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Final model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=31, random_state=42, verbose=-1))
])

# Fit and predict
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Save
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)

# Feature Importance
importances = model.named_steps['model'].feature_importances_
feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features))
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
print(feature_importance_df.sort_values('importance', ascending=False).head(5))
