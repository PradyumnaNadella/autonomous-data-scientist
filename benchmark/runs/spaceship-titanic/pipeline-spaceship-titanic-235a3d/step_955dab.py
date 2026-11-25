import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb
from features import build_features

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature processing check
train_feat = build_features(train)
print("Features created:", train_feat.columns.tolist())

# Check for leakage in features (did we include target in feature engineering? No.)
# Check for ID columns used as numbers
print("Are there numeric features that are actually IDs?", [col for col in train_feat.columns if 'Id' in col or 'Num' in col or 'Group' in col])

# Check experiments.jsonl
with open('experiments.jsonl', 'r') as f:
    for line in f:
        print(json.loads(line))
