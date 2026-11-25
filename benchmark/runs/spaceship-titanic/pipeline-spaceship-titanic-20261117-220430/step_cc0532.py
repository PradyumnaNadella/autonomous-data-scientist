# Tuning XGBoost
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define search space
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
