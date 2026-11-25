import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from features import build_features

train = pd.read_csv('train.csv')
train_f = build_features(train)
y = train_f['Transported']
X = train_f.drop(columns=['Transported', 'PassengerId'])

# Check for object types in original X
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)

cat_features = X.select_dtypes(include=['object']).columns.tolist()
num_features = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# Re-run CV to verify 0.8018 score
model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=5, n_estimators=200, random_state=42, verbose=-1)
pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
print(f"CV scores: {scores}")
print(f"Mean CV score: {scores.mean()}")
