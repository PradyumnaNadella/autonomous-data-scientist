# Tuning XGBoost
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Use the same setup as before
# ... (already defined in previous cells)
# Let's just re-run the tuning

param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
}

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42)
random_search.fit(train_df[features], train_df[target])

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
