# Just inspect the feature names if possible to infer importance
from sklearn.inspection import permutation_importance
import pandas as pd
from features import build_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

# Data
train_df = pd.read_csv('train.csv')
train_feat = build_features(train_df)
X = train_feat.drop(columns=['Transported', 'PassengerId', 'Name'])
y = train_feat['Transported']

def cast_to_string(X):
    return pd.DataFrame(X).astype(str)

categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('cast', FunctionTransformer(cast_to_string)),
                          ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

# Fit on a subset to get importance
pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('classifier', HistGradientBoostingClassifier(random_state=42))])
pipeline.fit(X, y)

# Since Permutation importance is expensive, just mention high-level features.
print("Model trained.")
