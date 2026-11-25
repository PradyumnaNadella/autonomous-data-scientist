from sklearn.model_selection import GridSearchCV

# XGBoost Tuning
xgb_model = Pipeline([('preprocessor', preprocessor), 
                      ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
