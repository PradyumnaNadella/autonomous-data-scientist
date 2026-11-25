import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Feature Engineering
train_feat = build_features(train_df)
test_feat = build_features(test_df)

# Drop non-feature columns
drop_cols = ['PassengerId', 'Name']
train_feat = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
test_feat = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])

X = train_feat.drop(columns=['Transported'])
y = train_feat['Transported']

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

def get_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    clf = get_pipeline(model)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name} CV Score: {results[name]:.4f}")
