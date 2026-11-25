import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# 1. Prepare Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Use build_features from provided file
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)
X_test = build_features(test_df)

# Define column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# 2. Experimentation
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "RandomForest": Pipeline([('rf', None)]) # Placeholder
}

# XGBoost CV
pipeline = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"XGBoost CV Score: {scores.mean():.4f}")
