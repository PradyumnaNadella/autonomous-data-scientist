from sklearn.ensemble import GradientBoostingClassifier

clf_gb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])

cv_scores_gb = cross_val_score(clf_gb, X, y, cv=5, scoring='accuracy')
print(f"GB CV Score: {cv_scores_gb.mean():.4f}")
