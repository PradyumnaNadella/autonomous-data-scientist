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
test_df = pd.read_csv('test.csv')

# Feature engineering
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported']
X_test = build_features(test_df)

# Drop raw IDs or unhelpful free-text
drop_cols = ['PassengerId', 'Name']
X = X.drop([c for c in drop_cols if c in X.columns], axis=1)
X_test = X_test.drop([c for c in drop_cols if c in X_test.columns], axis=1)

# Convert EVERYTHING to string if it's not numeric
# And actually force the conversion for ALL non-numeric columns
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    # Also verify that no column was somehow left as boolean
    if X[col].dtype == bool:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

# Double check dtypes
print(X.dtypes)

# Identify columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

# Define models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy').mean()
    print(f"{name}: {score:.4f}")
