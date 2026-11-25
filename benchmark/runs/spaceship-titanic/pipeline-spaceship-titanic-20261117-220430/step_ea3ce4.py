# Final script check
import joblib
import pandas as pd
model = joblib.load('model.joblib')
print(type(model))
