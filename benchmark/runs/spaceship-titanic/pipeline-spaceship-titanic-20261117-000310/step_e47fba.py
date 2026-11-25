import pandas as pd
import joblib
from features import build_features

model = joblib.load('best_model.joblib')
test_df = pd.read_csv('test.csv')
X_test = build_features(test_df.copy()).drop('PassengerId', axis=1)

predictions = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': predictions.astype(bool)
})
submission.to_csv('predictions.csv', index=False)

# Get feature importance
# The classifier is the second step in the pipeline
classifier = model.named_steps['classifier']
preprocessor = model.named_steps['preprocessor']

# Extract feature names
# Numeric
num_names = preprocessor.transformers_[0][2]
# Categorical - need to get them from OHE
ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
cat_names = ohe.get_feature_names_out(preprocessor.transformers_[1][2])

feature_names = list(num_names) + list(cat_names)
importances = classifier.feature_importances_

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
print(feature_importance_df.sort_values(by='importance', ascending=False).head(5))
