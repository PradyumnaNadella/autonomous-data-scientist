# Re-run Random Forest to get a clean baseline
from sklearn.model_selection import cross_val_score
rf_pipe = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
scores_rf = cross_val_score(rf_pipe, df.drop(columns=[target, 'PassengerId']), df[target], cv=5, scoring='accuracy')
print(f"RF CV: {np.mean(scores_rf):.4f}")
