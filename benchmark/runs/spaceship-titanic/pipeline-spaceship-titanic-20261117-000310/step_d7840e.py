from sklearn.model_selection import GridSearchCV

# Tune LGBM
pipeline = Pipeline([('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])
param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [5, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
grid.fit(X, y)
print(f"Best Score: {grid.best_score_:.4f}")
print(f"Best Params: {grid.best_params_}")
