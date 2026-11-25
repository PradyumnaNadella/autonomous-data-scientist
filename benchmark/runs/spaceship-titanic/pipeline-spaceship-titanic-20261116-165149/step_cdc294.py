# Tuning LightGBM moderately
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(random_state=42, verbose=-1))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__num_leaves': [31, 50]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best CV Accuracy: {grid.best_score_:.4f}")
print(f"Best Params: {grid.best_params_}")
