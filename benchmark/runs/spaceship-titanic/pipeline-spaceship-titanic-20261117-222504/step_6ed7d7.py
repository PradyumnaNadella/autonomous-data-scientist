from sklearn.model_selection import GridSearchCV

# LightGBM Tuning
pipe_lgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lgb.LGBMClassifier(random_state=42, verbose=-1))
])

params = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [5, 10]
}

grid = GridSearchCV(pipe_lgb, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)
print(f'Best Params: {grid.best_params_}')
print(f'Best CV Score: {grid.best_score_:.4f}')
