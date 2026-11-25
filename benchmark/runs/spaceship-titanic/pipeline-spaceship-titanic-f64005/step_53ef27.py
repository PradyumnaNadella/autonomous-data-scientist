import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

# Load and process
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Use provided build_features
train_feat = build_features(train)
test_feat = build_features(test)

# Identify feature types
# Assuming build_features keeps numerical and categorical
num_cols = train_feat.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [c for c in num_cols if c != 'Transported']
cat_cols = train_feat.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(n_estimators=100, verbose=-1)
}

# CV loop
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train_feat.drop('Transported', axis=1)
y = train_feat['Transported']

for name, model in models.items():
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
    
    # Placeholder for logging (we'll log in the next step)
    # The instructions say call log_experiment
