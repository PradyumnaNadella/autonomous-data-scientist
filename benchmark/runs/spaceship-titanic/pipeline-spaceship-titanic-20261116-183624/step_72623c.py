import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Load
df_train = pd.read_csv('train.csv')
X_train = build_features(df_train.drop('Transported', axis=1))
y_train = df_train['Transported'].astype(int)

# Fix types to be uniform: convert all columns to string
for col in X_train.columns:
    X_train[col] = X_train[col].astype(str)

# Re-identify types
# Actually, since everything is a string now, we need to be careful.
# Let's keep numeric-like features as numeric if possible, but safely.
# Let's rebuild properly.

# Let's reload and convert booleans to strings *before* build_features if possible,
# or just cast after build_features.
df_train = pd.read_csv('train.csv')
X_train = build_features(df_train.drop('Transported', axis=1))

# Convert everything to object/string to avoid OHE mixed type errors. 
# Or better: cast known numeric columns back to float and let OHE handle categoricals.
# Let's see the dtypes
print(X_train.dtypes)
