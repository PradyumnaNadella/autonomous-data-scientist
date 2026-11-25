import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data and the model
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
model = joblib.load('model.joblib')

# Check if model is a pipeline and contains preprocessing
print(f"Model type: {type(model)}")
# Inspect the pipeline structure
if isinstance(model, Pipeline):
    print("Pipeline steps:", model.named_steps.keys())
    preprocessor = model.named_steps.get('preprocessor')
    if preprocessor:
        print("Preprocessor type:", type(preprocessor))
    else:
        print("No 'preprocessor' step found. Checking for other steps...")

# Verify prediction format
preds = pd.read_csv('predictions.csv')
print("Predictions columns:", preds.columns)
print("Predictions shape:", preds.shape)
print("Test shape:", test.shape)

# Check for leakage by looking at the model's feature importance/preprocessing
# Simple check if the preprocessor was fitted on the full set
# (Usually we cannot verify this directly from the joblib unless we check the attributes)
