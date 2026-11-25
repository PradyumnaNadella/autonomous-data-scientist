# Create the train_model.py
script = """
import pandas as pd
import joblib
from features import build_features

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Build features
X = build_features(train_df.drop(columns=['Transported']))
y = train_df['Transported'].astype(int)
X_test = build_features(test_df)

# Ensure same types
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side']:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Load pipeline
pipe = joblib.load('model.joblib')

# Predictions
preds = pipe.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved.")
"""

with open('train_model.py', 'w') as f:
    f.write(script)
