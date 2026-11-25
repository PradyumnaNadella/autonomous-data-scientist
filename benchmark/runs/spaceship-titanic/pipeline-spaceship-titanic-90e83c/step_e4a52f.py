import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json

# Minimal run to confirm functionality
print("Pandas version:", pd.__version__)
train = pd.read_csv('train.csv')
print("Train shape:", train.shape)
