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

# The issue is likely in the build_features output returning mixed types even if we convert here.
# Let's inspect the types again right before splitting.
for col in X_processed.columns:
    if X_processed[col].dtype == 'object':
        X_processed[col] = X_processed[col].astype(str)
    elif X_processed[col].dtype == 'bool':
        X_processed[col] = X_processed[col].astype(str)

# Double check
print(f"X types check:\n{X_processed.dtypes}")

# Re-define categorical features to ONLY include what is definitely not numeric
numeric_features = X_processed.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
categorical_features = [c for c in X_processed.columns if c not in numeric_features]

print(f"Numeric columns: {numeric_features}")
print(f"Categorical columns: {categorical_features}")

# Define Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
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
