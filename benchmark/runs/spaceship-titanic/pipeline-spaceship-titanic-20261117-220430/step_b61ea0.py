import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

train = pd.read_csv('train.csv')
X = build_features(train.drop(columns=['Transported']))
y = train['Transported'].astype(int)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Tuning XGBoost
param_distributions = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

search = RandomizedSearchCV(xgb_pipeline, param_distributions, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
search.fit(X, y)

print(f"Best score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# Save the best model
joblib.dump(search.best_estimator_, 'model.joblib')
