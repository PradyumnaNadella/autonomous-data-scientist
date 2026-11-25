# Create train_model.py
with open('train_model.py', 'w') as f:
    f.write('''
import pandas as pd
import joblib
from features import build_features

# Load model and data
model = joblib.load('model.joblib')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Prepare data
X_test = build_features(test)

# Predict
predictions = model.predict(X_test)

# Save
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions.astype(bool)})
output.to_csv('predictions.csv', index=False)
''')
