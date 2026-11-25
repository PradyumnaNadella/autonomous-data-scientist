from features import build_features
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
df = pd.read_csv('train.csv')
X = df.drop('Transported', axis=1)
y = df['Transported'].astype(int)

# Use build_features
X = build_features(X)

# Define column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Models
rf = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
xgb = Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42))])

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
scores_xgb = cross_val_score(xgb, X, y, cv=cv, scoring='accuracy')

print(f"RF CV Score: {scores_rf.mean():.4f}")
print(f"XGB CV Score: {scores_xgb.mean():.4f}")
