from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Tuning XGB
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

param_dist = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1]
}

search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
search.fit(X, y)
print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")
