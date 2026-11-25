import joblib
import pandas as pd
from features import build_features

# Check how the model uses the features
model = joblib.load('model.joblib')
print(model.named_steps['preprocessor'])
# Inspect the transformers
transformers = model.named_steps['preprocessor'].transformers_
print("Transformers:", [t[0] for t in transformers])
