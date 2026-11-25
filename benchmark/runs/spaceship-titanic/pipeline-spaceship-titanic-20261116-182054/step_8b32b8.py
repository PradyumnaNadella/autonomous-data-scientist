import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from features import build_features

# 1. Load and prepare
df = pd.read_csv('train.csv')
X = df.drop('Transported', axis=1)
y = df['Transported']
X = build_features(X)

# Drop high cardinality/ID
to_drop = ['PassengerId', 'Name', 'Cabin'] # build_features handles Cabin? Let's check.
# Assuming build_features added new columns and we need to drop the old ones
for col in ['PassengerId', 'Name', 'Cabin']:
    if col in X.columns:
        X = X.drop(columns=[col])

# Categorical vs Numeric
cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 2. Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# 3. Models
clf_lr = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
clf_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100))])

# 4. CV
cv = StratifiedKFold(n_splits=5)
score_lr = cross_val_score(clf_lr, X, y, cv=cv, scoring='accuracy').mean()
score_rf = cross_val_score(clf_rf, X, y, cv=cv, scoring='accuracy').mean()

print(f"LR: {score_lr}")
print(f"RF: {score_rf}")
