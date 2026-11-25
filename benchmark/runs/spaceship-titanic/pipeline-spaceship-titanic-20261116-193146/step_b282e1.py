import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

# Load data
train_df = pd.read_csv('train.csv')

# Feature engineering
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported']

# Drop raw IDs
drop_cols = ['PassengerId', 'Name']
X = X.drop([c for c in drop_cols if c in X.columns], axis=1)

# Check types
print(X.dtypes)
# It seems some columns are still 'object' or 'bool' which might be the cause
# Let's inspect the first few rows
print(X.head())
