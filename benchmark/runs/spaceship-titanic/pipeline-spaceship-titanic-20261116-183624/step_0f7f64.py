from sklearn.model_selection import cross_val_score
# Experiment 1: RandomForest
rf_clf = Pipeline([('preprocessor', preprocessor), 
                   ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
rf_score = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"RandomForest: {rf_score:.4f}")

# Experiment 2: XGBoost (Default)
xgb_default_clf = Pipeline([('preprocessor', preprocessor), 
                           ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
xgb_default_score = cross_val_score(xgb_default_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"XGBoost (default): {xgb_default_score:.4f}")

# Experiment 3: XGBoost (Tuned)
xgb_tuned_clf = Pipeline([('preprocessor', preprocessor), 
                          ('model', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42))])
xgb_tuned_score = cross_val_score(xgb_tuned_clf, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"XGBoost (tuned): {xgb_tuned_score:.4f}")
