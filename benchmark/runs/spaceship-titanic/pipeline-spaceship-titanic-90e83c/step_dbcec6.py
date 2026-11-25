import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# A simple check if we can even run sklearn
print(f"Scikit-learn version: {sklearn.__version__}")

# Minimal script to ensure we can load data and compute something
train = pd.read_csv('train.csv')
print(f"Train rows: {len(train)}")
