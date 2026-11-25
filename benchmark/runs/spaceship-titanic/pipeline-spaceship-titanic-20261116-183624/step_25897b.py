import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import build_features

# Load data
train_df = pd.read_csv('train.csv')
X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int)

# Preprocessing
X_eng = build_features(X)
for col in X_eng.columns:
    if X_eng[col].dtype == 'object' or X_eng[col].dtype == 'bool':
        X_eng[col] = X_eng[col].astype(str)
drop_cols = ['PassengerId', 'Name']
X_eng = X_eng.drop(columns=[c for c in drop_cols if c in X_eng.columns])

numeric_features = X_eng.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_eng.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Tune XGBoost
clf = Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.1, 0.05]
}

grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_eng, y)
print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
