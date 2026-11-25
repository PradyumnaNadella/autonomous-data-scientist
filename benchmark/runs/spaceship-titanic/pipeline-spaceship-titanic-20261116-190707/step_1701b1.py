import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import build_features

train_df = pd.read_csv('train.csv')
X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported'].astype(int)

# Preprocessing
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(str).replace('nan', 'missing')
drop_cols = ['PassengerId', 'Surname', 'Group', 'Num']
X_train = X_train.drop(columns=drop_cols)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

xgb_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])
param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5, 7]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(xgb_pipe, param_grid, cv=cv, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best CV Score: {grid.best_score_}")
print(f"Best Params: {grid.best_params_}")
