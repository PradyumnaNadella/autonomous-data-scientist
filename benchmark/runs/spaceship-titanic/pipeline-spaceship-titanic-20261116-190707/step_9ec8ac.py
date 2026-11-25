# Tune LightGBM
from sklearn.model_selection import RandomizedSearchCV

pipe = Pipeline([('pre', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])
param_distributions = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [5, 10, -1]
}

search = RandomizedSearchCV(pipe, param_distributions, cv=5, scoring='accuracy', n_iter=5, n_jobs=-1, random_state=42)
search.fit(X, y)
print(f"Best score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")
log_experiment("LightGBM Tuned", search.best_score_, f"Params: {search.best_params_}")
