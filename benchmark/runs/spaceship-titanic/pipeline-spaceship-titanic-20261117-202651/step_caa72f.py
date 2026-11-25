from sklearn.model_selection import RandomizedSearchCV

# Define search space for LGBM
param_distributions = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [5, 10, 15]
}

lgbm_tuned = RandomizedSearchCV(
    lgbm, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)

lgbm_tuned.fit(X, y)
print(f"Best parameters: {lgbm_tuned.best_params_}")
print(f"Tuned LGBM CV Accuracy: {lgbm_tuned.best_score_:.4f}")
