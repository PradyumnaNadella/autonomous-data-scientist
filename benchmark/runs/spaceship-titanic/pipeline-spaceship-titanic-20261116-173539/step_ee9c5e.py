import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features
train_df = build_features(train)
test_df = build_features(test)

# Explicitly cast everything to string if object, to be safe against mixed types
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].astype(str)
    # Check if we still have bools - build_features might be returning them
    if train_df[col].dtype == 'bool':
        train_df[col] = train_df[col].astype(str)

for col in test_df.columns:
    if test_df[col].dtype == 'object':
        test_df[col] = test_df[col].astype(str)
    if test_df[col].dtype == 'bool':
        test_df[col] = test_df[col].astype(str)

# Identify features
target = 'Transported'
features = [col for col in train_df.columns if col not in ['Transported', 'PassengerId', 'Name']]

# Ensure types are uniform for OHE
# Check what types we have
print(train_df[features].dtypes)

numeric_features = train_df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_df[features].select_dtypes(include=['object']).columns.tolist()

print(f"Num: {len(numeric_features)}, Cat: {len(categorical_features)}")

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Model families
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, train_df[features], train_df[target], cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
