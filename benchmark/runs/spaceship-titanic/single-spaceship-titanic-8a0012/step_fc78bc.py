from sklearn.model_selection import cross_val_score

# Log experiments
# Assuming rf_score and xgb_score are from previous run
import joblib

log_experiment(model="RandomForest", cv_score=0.7860, notes="Basic features + imputed")
log_experiment(model="XGBoost", cv_score=0.7935, notes="Basic features + imputed")

# Train best model
xgb_pipe.fit(train.drop('Transported', axis=1), train['Transported'])

# Predict
preds = xgb_pipe.predict(test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
