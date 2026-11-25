import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print("Checking environment...")
try:
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
except Exception as e:
    print(e)
