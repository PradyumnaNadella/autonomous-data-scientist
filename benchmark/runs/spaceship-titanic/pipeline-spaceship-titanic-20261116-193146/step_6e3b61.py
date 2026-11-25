import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import build_features

# Load and process data
train_df = pd.read_csv('train.csv')
X = train_df.drop('Transported', axis=1)
y = train_df['Transported']

# Get features
X_processed = build_features(X)

# Explicitly cast boolean to str
for col in X_processed.select_dtypes(include=['bool']).columns:
    X_processed[col] = X_processed[col].astype(str)

# Define columns for preprocessing
numeric_features = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_processed.select_dtypes(include=['object']).columns.tolist()

# Define Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ]
)

# Define models
models = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=42))
]

# Evaluate models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models:
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X_processed, y, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
