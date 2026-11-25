import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from features import build_features

# Load and prepare data
train_df = pd.read_csv('train.csv')
X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int)

# Use build_features from provided script
X_eng = build_features(X)
# Drop IDs/Names as per rules
drop_cols = ['PassengerId', 'Name']
X_eng = X_eng.drop(columns=[c for c in drop_cols if c in X_eng.columns])

# Identify numeric and categorical columns
numeric_features = X_eng.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_eng.select_dtypes(include=['object', 'bool']).columns

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Models to compare
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X_eng, y, cv=skf, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
