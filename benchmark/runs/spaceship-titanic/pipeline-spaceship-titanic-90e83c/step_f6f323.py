import pandas as pd
import numpy as np

# I will write the function to a file named features.py so it can be imported.
features_code = """
import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['Num'] = cabin[1]
    df['Side'] = cabin[2]
    
    billing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[billing_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    
    df['Surname'] = df['Name'].str.split().str[-1]
    
    return df.drop(['Name', 'Cabin'], axis=1)
"""

with open('features.py', 'w') as f:
    f.write(features_code)

from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_f = build_features(train)
test_f = build_features(test)

X = train_f.drop(['PassengerId', 'Transported'], axis=1)
y = train_f['Transported'].astype(int)

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Evaluate models
models = [
    ("RandomForest", RandomForestClassifier(random_state=42)),
    ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ("LightGBM", LGBMClassifier(random_state=42, verbose=-1))
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    pipeline = Pipeline([('pre', preprocessor), ('clf', model)])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
    # We call log_experiment in the next steps, this is just to print
