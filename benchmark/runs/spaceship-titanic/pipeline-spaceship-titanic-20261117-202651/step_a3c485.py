# Log experiments as required
import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import sys

# The instructions say: "call log_experiment for each. This is a tool call, not a Python function"
# So I will output the data for the logs here and use the tool in the final turn.
# Oh wait, I cannot call a tool from inside Python.
# I need to print the values clearly for the modeler to use in the final step.
# Wait, "Call this once for every model you evaluate... This is a tool call, not a Python function".
# Okay, I will perform all required experiments here, and then use the `log_experiment` tool.
print("READY_FOR_LOGGING")
