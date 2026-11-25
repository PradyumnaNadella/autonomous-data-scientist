# Tuning RF
from sklearn.model_selection import RandomizedSearchCV

# Define pipeline for RF
clf_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

param_dist = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

search = RandomizedSearchCV(clf_rf, param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42)
search.fit(X, y)

print(f"Best RF Score: {search.best_score_}")
print(f"Best Params: {search.best_params_}")
