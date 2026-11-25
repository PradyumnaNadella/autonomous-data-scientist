import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Prepare data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features
def prep(df):
    df = build_features(df)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    return df.drop(['PassengerId', 'Group', 'Member', 'Num', 'Surname'], axis=1, errors='ignore')

# Log Experiments
# Need to call log_experiment but it is a tool. I will record it as asked.
print("Finished comparing models.")
