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

# Force conversions
for df in [X, X_test]:
    # Ensure no bool, convert to string
    for col in df.columns:
        if df[col].dtype == 'bool' or df[col].dtype == object:
            df[col] = df[col].astype(str)

# Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Define models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy').mean()
    print(f"{name}: {score:.4f}")
