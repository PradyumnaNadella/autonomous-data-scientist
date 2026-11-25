from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Tuning LGBM
lgbm = LGBMClassifier(random_state=42, verbose=-1)
params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7]
}

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm)])
random_search = RandomizedSearchCV(clf, params, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

random_search.fit(X, y)
print(f"Best Score: {random_search.best_score_:.4f}")
print(f"Best Params: {random_search.best_params_}")
