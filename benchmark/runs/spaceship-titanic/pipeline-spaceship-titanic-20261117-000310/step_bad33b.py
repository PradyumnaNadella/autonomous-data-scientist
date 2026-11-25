from sklearn.ensemble import GradientBoostingClassifier
model_gb = Pipeline([
    ('pre', preprocessor),
    ('clf', GradientBoostingClassifier(random_state=42))
])

scores_gb = cross_val_score(model_gb, train_df.drop(columns=['PassengerId', 'Transported']), train_df['Transported'], cv=5)
print(f"GradientBoosting Accuracy: {scores_gb.mean():.4f}")
