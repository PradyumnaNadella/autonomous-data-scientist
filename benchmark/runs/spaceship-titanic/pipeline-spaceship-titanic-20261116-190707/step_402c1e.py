import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Re-inspect model.joblib to check if it's a pipeline
model = joblib.load('model.joblib')
print(type(model))
if hasattr(model, 'steps'):
    print(model.steps)
else:
    print("Not a pipeline!")
