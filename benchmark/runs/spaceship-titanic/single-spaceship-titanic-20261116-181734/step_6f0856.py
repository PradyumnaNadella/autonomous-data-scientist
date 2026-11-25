# Train the best model (RF) on full dataset
from sklearn.ensemble import RandomForestClassifier

# Re-initialize the pipeline
final_model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

final_model.fit(X, y)

# Predict
predictions = final_model.predict(test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
