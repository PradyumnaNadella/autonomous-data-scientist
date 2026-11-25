# Re-run LightGBM with simple default to confirm
from features import build_features
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

train = pd.read_csv('train.csv')
train_feat = build_features(train)
y = train_feat['Transported'].astype(int)
X = train_feat.drop('Transported', axis=1)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Compare variants
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n in [100, 200]:
    lgbm = LGBMClassifier(n_estimators=n, verbose=-1)
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', lgbm)])
    score = cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()
    print(f"LGBM n_estimators={n}: {score:.4f}")
