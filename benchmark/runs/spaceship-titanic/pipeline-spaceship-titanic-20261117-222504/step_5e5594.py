import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from features import build_features

# Load and prep
train = pd.read_csv('train.csv')
train = build_features(train)
if 'PassengerId' in train.columns:
    train = train.drop(columns=['PassengerId'])

X = train.drop(columns=['Transported'])
y = train['Transported']

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)

cat_features = X.select_dtypes(include=['object']).columns.tolist()
num_features = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# LightGBM Tuning
pipe_lgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lgb.LGBMClassifier(random_state=42, verbose=-1))
])

params = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [5, 10]
}

grid = GridSearchCV(pipe_lgb, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)
print(f'Best Params: {grid.best_params_}')
print(f'Best CV Score: {grid.best_score_:.4f}')
