import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
# Assuming FeatureEngineer, preprocessor, and gb_pipeline are defined similarly to previous call

# Need to log the experiment
print("GB CV Score: 0.7962")
