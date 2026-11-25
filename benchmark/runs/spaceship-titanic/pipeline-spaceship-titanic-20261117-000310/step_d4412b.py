from sklearn.model_selection import GridSearchCV
# Hyperparameter tuning for LightGBM
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LGBMClassifier(random_state=42, verbose=-1))])
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [5, 10]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
