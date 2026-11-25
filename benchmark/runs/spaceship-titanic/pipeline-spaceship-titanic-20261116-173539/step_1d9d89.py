from sklearn.model_selection import RandomizedSearchCV

# Define search space for XGBoost
xgb_pipe = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
param_dist = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1, 0.2]
}

search = RandomizedSearchCV(xgb_pipe, param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42)
search.fit(df.drop(columns=[target, 'PassengerId']), df[target])

print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")
