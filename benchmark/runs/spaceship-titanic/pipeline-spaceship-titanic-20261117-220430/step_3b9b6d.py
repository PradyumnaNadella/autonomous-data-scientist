import joblib
model = joblib.load('model.joblib')
# The cat transformer is using one hot encoding on columns like "PassengerId", "Name", "Num"
# These are essentially unique IDs or high-cardinality features which is likely bad or overfitting.
# The user claims "No leakage". Let's check if the pipeline is actually fit on the full train data inside train_model.py
# (Actually, I should check the code that CREATES the model, not just train_model.py)
# Wait, there is no file that creates the model?
import os
print(os.listdir())
# Ah, I don't see the training script that generated 'model.joblib'.
