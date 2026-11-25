test = pd.read_csv('test.csv')
test_df = build_features(test)

# Fit model_gb on all training data
model_gb.fit(train_df.drop(columns=['PassengerId', 'Transported']), train_df['Transported'])

# Predict
predictions = model_gb.predict(test_df.drop(columns=['PassengerId']))

# Save predictions
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
