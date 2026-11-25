from features import build_features
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

# Log experiments
def log_experiment(cv_score, model, notes=""):
    print(f"Logged: {model} with score {cv_score:.4f} ({notes})")

# Load and process
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

# Run CV for experiments
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(n_estimators=100, verbose=-1)
}

for name, model in models.items():
    clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    log_experiment(scores.mean(), name)

# Tune LightGBM
tuned_lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
clf_tuned = Pipeline([('preprocessor', preprocessor), ('classifier', tuned_lgbm)])
scores_tuned = cross_val_score(clf_tuned, X, y, cv=cv, scoring='accuracy')
log_experiment(scores_tuned.mean(), "LightGBM, tuned", "n_estimators=300, learning_rate=0.05")

# Retrain and predict
clf_tuned.fit(X, y)
test = pd.read_csv('test.csv')
test_feat = build_features(test)
for col in test_feat.select_dtypes(include=['object']).columns:
    test_feat[col] = test_feat[col].astype(str)

preds = clf_tuned.predict(test_feat)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
