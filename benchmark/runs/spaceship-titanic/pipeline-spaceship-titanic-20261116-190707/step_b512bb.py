# Moderate tuning of XGBoost
from sklearn.model_selection import RandomizedSearchCV

xgb = get_pipeline(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))

param_dist = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 6, 9],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X, y)

print(f"Best Score: {random_search.best_score_}")
print(f"Best Params: {random_search.best_params_}")
